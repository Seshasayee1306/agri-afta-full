# fed_afta/client.py
import numpy as np
import xgboost as xgb
import torch
from fed_afta.utils import uncertainty_sampling_proba

class Client:
    def __init__(self, client_id, df, encoder, config):
        self.client_id = client_id
        self.df = df.reset_index(drop=True)
        self.encoder = encoder                # shared TabNet encoder
        self.config = config
        self.local_head = None               # XGBoost booster

    def _load_encoder_state(self, state):
        """
        Load global encoder state safely. Works with TabNet attribute name `network_`.
        """
        if state is None:
            return

        # Find the underlying network attribute
        net = getattr(self.encoder.clf, "network_", None)
        if net is None:
            # fallback: older naming
            net = getattr(self.encoder.clf, "network", None)

        if net is None:
            print(f"[Client {self.client_id}] Warning: Encoder has no network_( ) to load state into.")
            return

        try:
            net.load_state_dict(state)
        except Exception as e:
            print(f"[Client {self.client_id}] Error loading encoder state:", e)

    def _get_clean_state_dict(self):
        """
        Return a CPU-only state_dict for the encoder so the server can aggregate it.
        """
        net = getattr(self.encoder.clf, "network_", None)
        if net is None:
            net = getattr(self.encoder.clf, "network", None)
        if net is None:
            return None

        state = net.state_dict()
        clean = {}

        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                clean[k] = v.detach().cpu()
            else:
                clean[k] = torch.tensor(v)

        return clean

    def local_train(self, global_encoder_state=None):
        """
        Train the client encoder & head model:

        - Load global encoder parameters (if provided)
        - Generate embeddings
        - Select training samples via active learning (uncertainty sampling)
        - Train/Update local XGBoost head
        """

        # 1. Load global encoder params
        self._load_encoder_state(global_encoder_state)

        # 2. Prepare data
        features = self.config['features']
        target = self.config['target']

        X = self.df[features].fillna(0).values
        y = self.df[target].values

        # Extract embeddings from TabNet
        emb = self.encoder.get_embeddings(X)

        # 3. Active learning or initial sampling
        if self.local_head is not None:
            # Use previous head for uncertainty sampling
            try:
                d = xgb.DMatrix(emb)
                probs = self.local_head.predict(d)

                k = min(self.config.get("active_k", 200), len(emb))
                idx = uncertainty_sampling_proba(probs, k=k)

                X_train = emb[idx]
                y_train = y[idx]

            except Exception:
                # fallback if booster fails
                ntrain = min(2000, len(emb))
                idx = np.random.choice(len(emb), size=ntrain, replace=False)
                X_train = emb[idx]
                y_train = y[idx]

        else:
            # Initial training subset
            ntrain = min(2000, len(emb))
            idx = np.random.choice(len(emb), size=ntrain, replace=False)
            X_train = emb[idx]
            y_train = y[idx]

        # 4. Train XGBoost head
        dtrain = xgb.DMatrix(X_train, label=y_train)
        params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'eta': 0.05,
            'verbosity': 0
        }

        bst = xgb.train(params, dtrain, num_boost_round=200)
        self.local_head = bst

        # 5. Return (encoder_state_dict, xgboost_model)
        return self._get_clean_state_dict(), self.local_head