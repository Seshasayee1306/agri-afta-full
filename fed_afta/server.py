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
    - Evaluates and saves the final artifact to backend/final_model.pkl (absolute path)
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
            self.clients[cid] = Client(cid, cdf, self.encoder, self.config)

    def create_public_val(self, frac=0.02):
        """Create a small public validation set for distillation."""
        if frac <= 0 or frac >= 1:
            frac = 0.02
        self.public_val = self.df_all.sample(frac=frac, random_state=42).reset_index(drop=True)
        print(f"[Server] Public validation set created: {len(self.public_val)} rows")

    def _get_encoder_state(self):
        """
        Safely return a CPU-only state_dict for the encoder (if available).
        Looks for encoder.clf.network_ (typical PyTorch style) then encoder.clf.network,
        then encoder.clf.state_dict() as fallback.
        Returns None if no state is available.
        """
        # Preferred attribute: network_
        net = getattr(self.encoder.clf, "network_", None)
        if net is not None:
            sd = net.state_dict()
        else:
            # fallback to network or direct state_dict
            net2 = getattr(self.encoder.clf, "network", None)
            if net2 is not None:
                sd = net2.state_dict()
            else:
                # fallback to encoder.clf.state_dict() if available
                try:
                    sd = self.encoder.clf.state_dict()
                except Exception:
                    return None

        # ensure CPU tensors
        clean = {}
        for k, v in sd.items():
            if isinstance(v, torch.Tensor):
                clean[k] = v.detach().cpu()
            else:
                try:
                    # convert numpy-like to tensor
                    clean[k] = torch.tensor(v)
                except Exception:
                    # last resort: skip
                    continue
        return clean if len(clean) > 0 else None

    def aggregate_encoder(self, state_dicts, weights):
        """
        Weighted average of encoder state dicts.
        state_dicts: list of dicts (torch tensors or arrays)
        weights: list of ints (client sizes)
        """
        if not state_dicts:
            print("[Server] No state_dicts provided to aggregate.")
            return

        # determine union of keys
        all_keys = set()
        for sd in state_dicts:
            all_keys.update(sd.keys())
        total_weight = float(sum(weights)) if sum(weights) > 0 else float(len(weights))

        averaged = {}
        for k in all_keys:
            accum = None
            for sd, w in zip(state_dicts, weights):
                if k not in sd:
                    continue
                v = sd[k]
                # convert to numpy array
                if isinstance(v, torch.Tensor):
                    arr = v.detach().cpu().numpy()
                elif isinstance(v, np.ndarray):
                    arr = v
                else:
                    try:
                        arr = np.array(v)
                    except Exception:
                        raise RuntimeError(f"Unsupported tensor type for key {k}: {type(v)}")
                if accum is None:
                    accum = arr * float(w)
                else:
                    accum += arr * float(w)
            if accum is None:
                continue
            avg = accum / total_weight
            averaged[k] = torch.tensor(avg)

        # attempt to load averaged weights back into encoder
        net_attr = getattr(self.encoder.clf, "network_", None) or getattr(self.encoder.clf, "network", None)

        if net_attr is not None:
            try:
                net_attr.load_state_dict(averaged, strict=False)
                print("[Server] Loaded averaged state into encoder.network_ (or network).")
            except Exception as e:
                raise RuntimeError("Failed to load averaged state into encoder network: " + str(e))
        else:
            # fallback: try encoder.clf.load_state_dict
            try:
                self.encoder.clf.load_state_dict(averaged, strict=False)
                print("[Server] Loaded averaged state via encoder.clf.load_state_dict.")
            except Exception as e:
                raise RuntimeError("Could not load averaged encoder state into encoder object: " + str(e))

    def distill_heads(self, client_heads):
        """
        Distill client xgboost heads by averaging their predicted probabilities on a public set,
        generating pseudo-labels, and training a new global XGBoost head on embeddings vs pseudo-labels.
        """
        if self.public_val is None:
            raise RuntimeError("Public validation set not created. Call create_public_val() first.")

        features = self.config['features']
        Xpub = self.public_val[features].fillna(0).values
        emb = self.encoder.get_embeddings(Xpub)

        prob_list = []
        for head in client_heads:
            try:
                dpub = xgb.DMatrix(emb)
                p = head.predict(dpub)
                prob_list.append(p)
            except Exception as e:
                print(f"[Server] Warning: skipping a client head due to predict error: {e}")

        if len(prob_list) == 0:
            raise RuntimeError("No client head predictions available for distillation.")

        probs = np.vstack(prob_list)  # shape (n_clients, n_samples)
        avg_prob = probs.mean(axis=0)
        pseudo = (avg_prob >= 0.5).astype(int)

        dtrain = xgb.DMatrix(emb, label=pseudo)
        params = {'objective': 'binary:logistic', 'max_depth': 6, 'eta': 0.05, 'verbosity': 0}
        self.global_head = xgb.train(params, dtrain, num_boost_round=300)
        print("[Server] Trained distilled global head on pseudo-labels.")

    def run_rounds(self, rounds=5):
        """Run the federated rounds and save the final artifact."""
        self.create_public_val(frac=0.02)

        for r in range(1, rounds + 1):
            print(f"\n===== START ROUND {r}/{rounds} =====")
            state_list = []
            heads = []
            weights = []

            # provide current encoder state (if available) to clients
            global_state = self._get_encoder_state()
            if global_state is None:
                print("[Server] No encoder state available to send to clients (likely first round).")
            else:
                print("[Server] Sending global encoder state to clients.")

            # collect client updates
            for cid, client in self.clients.items():
                try:
                    state, head = client.local_train(global_encoder_state=global_state)
                except Exception as e:
                    print(f"[Server] Error during client {cid} local_train: {e}")
                    state, head = None, None

                if state is not None:
                    state_list.append(state)
                if head is not None:
                    heads.append(head)
                weights.append(len(client.df))

            # aggregate encoder weights if available
            if len(state_list) > 0:
                try:
                    self.aggregate_encoder(state_list, weights)
                except Exception as e:
                    raise RuntimeError("Failed to aggregate encoder states: " + str(e))
            else:
                print("[Server] Warning: no encoder states returned by clients; skipping encoder aggregation.")

            # distill heads if available
            if len(heads) > 0:
                try:
                    self.distill_heads(heads)
                except Exception as e:
                    raise RuntimeError("Failed to distill client heads: " + str(e))
            else:
                print("[Server] Warning: no client heads available; global head remains unchanged.")

            # evaluate global head on full dataset
            if self.global_head is not None:
                features = self.config['features']
                X_all = self.df_all[features].fillna(0).values
                emb_all = self.encoder.get_embeddings(X_all)
                d = xgb.DMatrix(emb_all)
                preds = (self.global_head.predict(d) >= 0.5).astype(int)
                acc = evaluate_preds(self.df_all[self.config['target']].values, preds)
                print(f"[Server] Global accuracy after round {r}: {acc:.4f}")
            else:
                print("[Server] No global head available yet to evaluate.")

        # Save final artifact (absolute path, ensure backend dir exists)
        base_dir = os.path.dirname(os.path.abspath(__file__))    # fed_afta/
        outpath = os.path.join(base_dir, "..", "backend", "final_model.pkl")
        outpath = os.path.abspath(outpath)
        os.makedirs(os.path.dirname(outpath), exist_ok=True)

        artifact = {
            'encoder_state': self._get_encoder_state(),
            'head': self.global_head
        }

        joblib.dump(artifact, outpath)
        print("[Server] Saved final model artifact to:", outpath)
