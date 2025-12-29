# fed_afta/server.py
import os
import joblib
import numpy as np
import xgboost as xgb
import torch
from fed_afta.utils import evaluate_preds


class Server:
    """
    Robust AFTA-style federated server:
    - Collects encoder state dicts and xgboost heads from clients
    - Weighted-averages encoder parameters
    - Distills client heads into a global xgboost head using a small public set
    - Evaluates and saves the final artifact to backend/final_model.pkl
    """

    def __init__(self, df_all, encoder, config):
        self.df_all = df_all.reset_index(drop=True)
        self.encoder = encoder
        self.config = config
        self.clients = {}
        self.global_head = None
        self.public_val = None

    def register_clients(self, client_dfs):
        """Register Client instances (dict: client_id -> dataframe)."""
        from fed_afta.client import Client
        for cid, cdf in client_dfs.items():
            self.clients[cid] = Client(str[cid], cdf, self.encoder, self.config)

    def create_public_val(self, frac=0.02):
        """Create a small public validation set for distillation."""
        if frac <= 0 or frac >= 1:
            frac = 0.02
        self.public_val = self.df_all.sample(frac=frac, random_state=42).reset_index(drop=True)
        print(f"[Server] Public validation set created: {len(self.public_val)} rows")

    def _get_encoder_state(self):
        """
        Safely return a CPU-only state_dict for the encoder (if available).
        """
        net = getattr(self.encoder.clf, "network_", None)
        if net is not None:
            sd = net.state_dict()
        else:
            net2 = getattr(self.encoder.clf, "network", None)
            if net2 is not None:
                sd = net2.state_dict()
            else:
                try:
                    sd = self.encoder.clf.state_dict()
                except Exception:
                    return None

        clean = {}
        for k, v in sd.items():
            if isinstance(v, torch.Tensor):
                clean[k] = v.detach().cpu()
        return clean if clean else None

    def aggregate_encoder(self, state_dicts, weights):
        """
        Weighted average of encoder state dicts.
        """
        if not state_dicts:
            print("[Server] No state_dicts provided to aggregate.")
            return

        all_keys = set()
        for sd in state_dicts:
            all_keys.update(sd.keys())

        total_weight = float(sum(weights)) if sum(weights) > 0 else float(len(weights))
        averaged = {}

        for k in all_keys:
            acc = None
            for sd, w in zip(state_dicts, weights):
                if k not in sd:
                    continue
                arr = sd[k].detach().cpu().numpy()
                acc = arr * w if acc is None else acc + arr * w
            averaged[k] = torch.tensor(acc / total_weight)

        net = getattr(self.encoder.clf, "network_", None) or getattr(self.encoder.clf, "network", None)
        if net is not None:
            net.load_state_dict(averaged, strict=False)
            print("[Server] Loaded averaged encoder state into network.")
        else:
            self.encoder.clf.load_state_dict(averaged, strict=False)
            print("[Server] Loaded averaged encoder state via clf.")

    def distill_heads(self, client_heads):
        """
        Distill client xgboost heads into a global head.
        """
        if self.public_val is None:
            raise RuntimeError("Public validation set not created.")

        features = self.config['features']
        Xpub = self.public_val[features].fillna(0).values
        emb = self.encoder.get_embeddings(Xpub)

        prob_list = []
        dpub = xgb.DMatrix(emb)

        for head in client_heads:
            try:
                p = head.predict(dpub)
                prob_list.append(p)
            except Exception as e:
                print(f"[Server] Warning: skipping a client head due to predict error: {e}")

        if len(prob_list) == 0:
            raise RuntimeError("No client head predictions available for distillation.")

        probs = np.vstack(prob_list)
        avg_prob = probs.mean(axis=0)
        pseudo = (avg_prob >= 0.5).astype(int)

        # -------------------------------------------------
        # GUARD PSEUDO-LABELS (NO LOGIC CHANGE)
        # -------------------------------------------------
        mask = np.isfinite(pseudo)
        emb = emb[mask]
        pseudo = pseudo[mask]

        if len(pseudo) == 0:
            raise ValueError("All pseudo-labels invalid; aborting round")

        dtrain = xgb.DMatrix(emb, label=pseudo)

        params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'eta': 0.05,
            'base_score': 0.5,   # 🔒 HARD LOCK (CRITICAL FIX)
            'verbosity': 0
        }

        self.global_head = xgb.train(params, dtrain, num_boost_round=300)
        print("[Server] Trained distilled global head.")

    def run_rounds(self, rounds=5):
        """Run federated rounds and save final artifact."""

        # -------------------------------------------------
        # FEATURE SANITIZATION (CRITICAL, NON-DESTRUCTIVE)
        # -------------------------------------------------
        feature_cols = self.config["features"]

        before_feat = len(self.df_all)
        self.df_all = self.df_all.replace([np.inf, -np.inf], np.nan)
        self.df_all = self.df_all.dropna(subset=feature_cols)

        print("[Server] Feature sanitization:")
        print(f" - Rows before: {before_feat}")
        print(f" - Rows after : {len(self.df_all)}")

        if len(self.df_all) == 0:
            raise ValueError("Dataset empty after feature sanitization")

        self.create_public_val(frac=0.02)

        for r in range(1, rounds + 1):
            print(f"\n===== START ROUND {r}/{rounds} =====")

            state_list = []
            heads = []
            weights = []

            global_state = self._get_encoder_state()

            for cid, client in self.clients.items():
                try:
                    state, head = client.local_train(global_encoder_state=global_state)
                except Exception as e:
                    print(f"[Server] Error during client {cid} local_train: {e}")
                    continue

                if state is not None:
                    state_list.append(state)
                if head is not None:
                    heads.append(head)
                weights.append(len(client.df))

            if len(state_list) > 0:
                self.aggregate_encoder(state_list, weights)
            else:
                print("[Server] Warning: no encoder states returned.")

            if len(heads) > 0:
                self.distill_heads(heads)
            else:
                print("[Server] Warning: no client heads returned.")

            if self.global_head is not None:
                X_all = self.df_all[self.config['features']].fillna(0).values
                emb_all = self.encoder.get_embeddings(X_all)
                d = xgb.DMatrix(emb_all)
                preds = (self.global_head.predict(d) >= 0.5).astype(int)
                acc = evaluate_preds(self.df_all[self.config['target']].values, preds)
                print(f"[Server] Global accuracy after round {r}: {acc:.4f}")
            else:
                print("[Server] No global head available for evaluation.")

        # -------------------------------------------------
        # SAVE FINAL ARTIFACT
        # -------------------------------------------------
        base_dir = os.path.dirname(os.path.abspath(__file__))
        outpath = os.path.abspath(os.path.join(base_dir, "..", "backend", "final_model.pkl"))
        os.makedirs(os.path.dirname(outpath), exist_ok=True)

        artifact = {
            'encoder_state': self._get_encoder_state(),
            'head': self.global_head
        }

        joblib.dump(artifact, outpath)
        print("[Server] Saved final model artifact to:", outpath)
