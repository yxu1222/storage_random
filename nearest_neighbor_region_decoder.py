import sys
sys.path.append("/u/yxu13/celltype_ibl-multiunit0/src")

import torch
import torch.nn as nn

from celltype_ibl.models.BiModalEmbedding import (
    BimodalEmbeddingModel,
)


from npyx.c4.dl_utils import (
    load_waveform_encoder,
)
from npyx.c4.dataset_init import BIN_SIZE, WIN_SIZE
from celltype_ibl.utils.ibl_label_ratio_MLP import ibl_representation_MLP_classifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    classification_report,
)
from joblib import load
import glob
import numpy as np
import pandas as pd
import pickle
import os
import pdb
import gc

WVF_ENCODER_ARGS_SINGLE = {
    "beta": 5,
    "d_latent": 10,
    "dropout_l0": 0.1,
    "dropout_l1": 0.1,
    "lr": 5e-5,
    "n_layers": 2,
    "n_units_l0": 600,
    "n_units_l1": 300,
    "optimizer": "Adam",
    "batch_size": 128,
}  # save in a separate file to import


model_dic = {'tempF_lnormF_dim512_simadjF_l2normT_batch1024_actgelu_data_ibl_seed42_date2024-06-17_rootF_heldoutT_init_f5bcff43': 100,
             'tempF_lnormF_dim512_simadjF_l2normT_batch1024_actgelu_data_ibl_seed43_date2024-06-17_rootF_heldoutT_init_e636e2dc': 100,
             'tempF_lnormF_dim512_simadjF_l2normT_batch1024_actgelu_data_ibl_seed44_date2024-06-17_rootF_heldoutT_init_c1c73f64': 100,
             'tempF_lnormF_dim512_simadjF_l2normT_batch1024_actgelu_data_ibl_seed45_date2024-06-17_rootF_heldoutT_init_829c666c': 100,
             'tempF_lnormF_dim512_simadjF_l2normT_batch1024_actgelu_data_ibl_seed46_date2024-06-17_rootF_heldoutT_init_739b4e86': 100}


# MODEL_PATH = "/mnt/sdceph/users/hyu10/cell-type_representation/contrastive_experiment/tempF_dim512_augT_l2normT_batch1024_actgelu_data_ibl_seed42_date2024-04-25-18-14-39_rootF_heldoutT_init/checkpoint_epoch_1770.pt"
MODEL_PATH = "/mnt/sdceph/users/hyu10/cell-type_representation/contrastive_experiment/tempF_dim512_augT_batch1024_actgelu_data_ibl_seed42_date2024-04-30-10-17-30_rootF_heldoutT_init/checkpoint_epoch_500.pt"
VAE_WVF_PATH = "/mnt/sdceph/users/hyu10/cell-type_representation/ibl_vaes/wvf_singlechannel_encoder.pt"
VAE_ACG_PATH = "/mnt/sdceph/users/hyu10/cell-type_representation/ibl_vaes/3DACG_logscale_encoder_gelu.pt"

SWEEP_DIR = "/projects/bcxj/yxu13/ibl_mwvf/cell-type_representation/tune_ibl_label_ratio_sweep"
LIN_DIR = "/projects/bcxj/yxu13/ibl_mwvf/cell-type_representation/linear_results/"
CLIP_MODEL_DIR = (
    "/projects/bcxj/yxu13/ibl_mwvf/cell-type_representation/contrastive_experiment/"
)
VAE_DIR = "/mnt/sdceph/users/hyu10/cell-type_representation/ibl_vaes"
SIMCLR_ACG_DIR = (
    "/mnt/sdceph/users/hyu10/cell-type_representation/SimCLR/ACG_chosen_by_acc"
)
SIMCLR_WVF_DIR = (
    "/mnt/sdceph/users/hyu10/cell-type_representation/SimCLR/WVF_chosen_by_acc"
)

def encode_ibl_depth_test_data(
    model_path=None,
    acg_path=None,
    wvf_path=None,
    test_fold=[3, 6],
    latent_dim=512,
    use_raw=False,
    embedding_model="contrastive",
    per_depth=True,
    return_depth=False,
):
    n_channels=25
    if use_raw:
        if per_depth:
            test_wvf_rep = np.load("/projects/bcxj/yxu13/ibl_mwvf/cell-type_representation/data/testing_5nn_wvfs_{}.npy".format(n_channels))
            test_acg_rep = np.load("/projects/bcxj/yxu13/ibl_mwvf/cell-type_representation/data/testing_5nn_acgs.npy")
            test_cosmos_region = np.load("/projects/bcxj/yxu13/ibl_mwvf/cell-type_representation/data/testing_5nn_labels.npy")
            return test_wvf_rep.reshape(-1,2250), test_acg_rep.reshape(-1,10,101), test_cosmos_region
        else:
            test_wvf_rep = np.load("/projects/bcxj/yxu13/ibl_mwvf/cell-type_representation/data/testing_wvfs_{}.npy".format(n_channels))
            test_acg_rep = np.load("/projects/bcxj/yxu13/ibl_mwvf/cell-type_representation/data/testing_acgs.npy")
            test_cosmos_region = np.load("/projects/bcxj/yxu13/ibl_mwvf/cell-type_representation/data/testing_labels.npy")
            return test_wvf_rep, test_acg_rep, test_cosmos_region
    else:
        if per_depth:
            test_wvf_rep = np.load("/projects/bcxj/yxu13/ibl_mwvf/cell-type_representation/data/testing_5nn_wvfs_{}.npy".format(n_channels))
            test_acg_rep = np.load("/projects/bcxj/yxu13/ibl_mwvf/cell-type_representation/data/testing_5nn_acgs.npy")
            test_cosmos_region = np.load("/projects/bcxj/yxu13/ibl_mwvf/cell-type_representation/data/testing_5nn_labels.npy")
            test_wvf_rep = test_wvf_rep.reshape(-1,2250)
            test_acg_rep = test_acg_rep.reshape(-1,10,101)
        else:
            test_wvf_rep = np.load("/projects/bcxj/yxu13/ibl_mwvf/cell-type_representation/data/testing_wvfs_{}.npy".format(n_channels))
            test_acg_rep = np.load("/projects/bcxj/yxu13/ibl_mwvf/cell-type_representation/data/testing_acgs.npy")
            test_cosmos_region = np.load("/projects/bcxj/yxu13/ibl_mwvf/cell-type_representation/data/testing_labels.npy")
        
        encode_model = BimodalEmbeddingModel(
            layer_norm=False,
            latent_dim=latent_dim,
            l2_norm=True,
            activation="gelu",
            batch_norm=False,
            in_features=90*25,
        )        

        
        checkpoint = torch.load(model_path)
        encode_model.load_state_dict(checkpoint["model_state_dict"])
        encode_model.eval()

        test_wvf = torch.tensor(test_wvf_rep.astype("float32"))
        test_acg = torch.tensor(test_acg_rep.astype("float32"))
        
        test_wvf_rep, test_acg_rep = encode_model.representation(
            test_wvf,
            test_acg.reshape(-1, 1, 10, 101) * 10,
        )
        
        test_wvf_rep = test_wvf_rep.detach().cpu().numpy()
        test_acg_rep = test_acg_rep.detach().cpu().numpy()

        return test_wvf_rep, test_acg_rep, test_cosmos_region
    '''# Load the logistic regression logits
    if (embedding_model == "contrastive") | (embedding_model == "supervise"):
        encode_model = BimodalEmbeddingModel(
            layer_norm=False,
            latent_dim=latent_dim,
            l2_norm=True,
            activation="gelu",
            batch_norm=False,
            in_features=90*25,
        )
            

        if not use_raw:
            checkpoint = torch.load(model_path)
            encode_model.load_state_dict(checkpoint["model_state_dict"])
            encode_model.eval()
    else:
        raise ValueError("Model not recognised")

    if per_depth:
        depth_wvf, depth_acg, depth_cosmos_region, depth_fold_idx, depth, pids = (
            get_ibl_wvf_acg_per_depth(return_region="Cosmos", return_depth=return_depth)
        )

    test_idx = [
        index
        for index, element in enumerate(depth_fold_idx)
        if (element in test_fold)
        and (depth_cosmos_region[index] != "void")
        and (depth_cosmos_region[index] != "root")
    ]
    test_wvf = depth_wvf[test_idx].reshape(-1, 90)
    test_acg = depth_acg[test_idx].reshape(-1, 10, 101)
    if use_raw:
        test_wvf_rep = test_wvf
        test_acg_rep = test_acg.reshape(-1, 10 * 101)
    else:
        test_wvf = torch.tensor(test_wvf.astype("float32"))
        test_acg = torch.tensor(test_acg.astype("float32"))

    test_cosmos_region = depth_cosmos_region[test_idx]

    if not use_raw:
        encode_model.eval()
        if (embedding_model == "contrastive") | (embedding_model == "simclr"):
            # Get the representations for test and training data
            test_wvf_rep, test_acg_rep = encode_model.representation(
                test_wvf,
                test_acg.reshape(-1, 1, 10, 101) * 10,
            )
        elif embedding_model == "vae":
            test_wvf_rep, test_acg_rep = encode_model.embed(
                test_wvf,
                test_acg.reshape(-1, 1, 10, 101) * 10,
                return_pre_projection=True,
            )
        test_wvf_rep = test_wvf_rep.detach().cpu().numpy()
        test_acg_rep = test_acg_rep.detach().cpu().numpy()

    if per_depth and return_depth:
        return test_wvf_rep, test_acg_rep, test_cosmos_region, depth[test_idx], pids[test_idx]
    else:
        return test_wvf_rep, test_acg_rep, test_cosmos_region
'''

def load_from_existing_model(
    embedding_model: str = "contrastive",
    freeze: bool = False,
    latent_dim: int = 512,
    seed: int = 42,
):
    if (
        (embedding_model == "contrastive")
        | (embedding_model == "supervise")
        | (embedding_model == "simclr")
    ):
        if embedding_model == "simclr":
            ln = True
        else:
            ln = False
        encode_model = BimodalEmbeddingModel(
            layer_norm=ln,
            latent_dim=latent_dim,
            l2_norm=True,
            activation="gelu",
            batch_norm=False,
            in_features=90*25,
        )
        num = 2
        if (embedding_model == "contrastive") | (embedding_model == "simclr"):
            num = 3
        
        globarr=glob.glob(SWEEP_DIR+"/**/*.pt")
        for i in model_dic.keys():
            if "seed"+str(seed) not in i:
                continue
            for j in globarr:
                if not freeze:
                    if i in j and "freeze" not in j:
                        best_checkpoint_path = j
                else:
                    if i in j and "freeze" in j:
                        best_checkpoint_path = j
    elif embedding_model == "vae":
        acg_vae = load_acg_vae(
            None,
            WIN_SIZE // 2,
            BIN_SIZE,
            initialise=False,
            pool="avg",
            activation="gelu",
        )
        acg_head = VAEEncoder(acg_vae.encoder.to("cpu"), 10)  # maybe change this?

        wvf_vae = load_waveform_encoder(
            WVF_ENCODER_ARGS_SINGLE,
            None,
            in_features=90,
            initialise=False,
        )
        wvf_head = VAEEncoder(wvf_vae.encoder, WVF_ENCODER_ARGS_SINGLE["d_latent"])

        encode_model = vae_encode_model(wvf_head, acg_head)

        if freeze:
            best_checkpoint_path = (
                SWEEP_DIR + "/vae2_seed_" + str(seed) + "freeze/best_model_1_00.pt"
            )
        else:
            best_checkpoint_path = (
                SWEEP_DIR + "/vae2_seed_" + str(seed) + "/best_model_1_00.pt"
            )
    else:
        raise ValueError("Model not recognised")
    if freeze:
        clf = ibl_representation_MLP_classifier(
            encode_model,
            embedding_model=embedding_model,
            n_classes=10,
            freeze_encoder_weights=freeze,
            modality="both",
            layer_size=[512,512,512],
            drop_out=0.1
        )
    else:
        clf = ibl_representation_MLP_classifier(
            encode_model,
            embedding_model=embedding_model,
            n_classes=10,
            freeze_encoder_weights=freeze,
            modality="both",
            layer_size=[512],
            drop_out=0.1
        )
    best_checkpoint = torch.load(best_checkpoint_path)
    clf.load_state_dict(best_checkpoint["model_state_dict"])

    return clf


def load_from_existing_logistic(
    embedding_model: str = "contrastive",
    seed: int = 42,
):
    dic = {'tempF_lnormF_dim512_simadjF_l2normT_batch1024_actgelu_data_ibl_seed42_date2024-06-17_rootF_heldoutT_init_f5bcff43': 100,
           'tempF_lnormF_dim512_simadjF_l2normT_batch1024_actgelu_data_ibl_seed43_date2024-06-17_rootF_heldoutT_init_e636e2dc': 100,
           'tempF_lnormF_dim512_simadjF_l2normT_batch1024_actgelu_data_ibl_seed44_date2024-06-17_rootF_heldoutT_init_c1c73f64': 100,
           'tempF_lnormF_dim512_simadjF_l2normT_batch1024_actgelu_data_ibl_seed45_date2024-06-17_rootF_heldoutT_init_829c666c': 100,
           'tempF_lnormF_dim512_simadjF_l2normT_batch1024_actgelu_data_ibl_seed46_date2024-06-17_rootF_heldoutT_init_739b4e86': 100}
    for model in dic.keys():
        if "seed"+str(seed) in model:
            break
    # Construct the filename based on how you saved it
    filename = f"{LIN_DIR}/{model}/linear_probe_best_F1/model.joblib"

    # Load the model
    clf_loaded = load(filename)
    return clf_loaded


def nearest_neighbor_logits_combine(
    model_path=MODEL_PATH,
    acg_path=VAE_ACG_PATH,
    wvf_path=VAE_WVF_PATH,
    model: str = "linear",
    test_fold: list[int] = [3, 6],
    latent_dim: int = 512,
    encoder: str = "contrastive",
    per_depth: bool = True,  # whether to use single unit or k nearest neighbors per depth
    freeze: bool = False,
    seed: int = 42,
    return_depth: bool = False
):
    """
    Combine the logits from different folds
    """
    if (model == "linear") and (encoder != "supervise"):
        use_raw = False
    else:
        use_raw = True
    if encoder == "contrastive":
        for modelaaaaa in model_dic.keys():
            if "seed"+str(seed) in modelaaaaa:
                model_path=CLIP_MODEL_DIR + f"/{modelaaaaa}/checkpoint_epoch_{model_dic[modelaaaaa]}.pt"
                break
    elif encoder == "vae":
        acg_path = VAE_DIR + f"/3DACG_logscale_seed_{seed}_encoder_gelu.pt"
        wvf_path = VAE_DIR + f"/wvf_singlechannel_seed_{seed}_encoder.pt"
    elif encoder == "simclr":
        simclr_wvf_patttern = SIMCLR_WVF_DIR + f"/checkpoint_acc_{seed}_epoch_*.pt"
        wvf_path = glob.glob(simclr_wvf_patttern)[0]

        simclr_acg_patttern = SIMCLR_ACG_DIR + f"/checkpoint_acc_{seed}_epoch_*.pt"
        acg_path = glob.glob(simclr_acg_patttern)[0]
    
    if return_depth: 
        test_wvf_rep, test_acg_rep, test_cosmos_region, depth, pids = encode_ibl_depth_test_data(
            model_path=model_path,
            acg_path=acg_path,
            wvf_path=wvf_path,
            test_fold=test_fold,
            latent_dim=latent_dim,
            use_raw=use_raw,
            embedding_model=encoder,
            per_depth=per_depth,
            return_depth = True
        )
    else:
        test_wvf_rep, test_acg_rep, test_cosmos_region = encode_ibl_depth_test_data(
            model_path=model_path,
            acg_path=acg_path,
            wvf_path=wvf_path,
            test_fold=test_fold,
            latent_dim=latent_dim,
            use_raw=use_raw,
            embedding_model=encoder,
            per_depth=per_depth,
        )

    N_test = len(test_cosmos_region)

    nan_idx = np.where(np.all(np.isnan(test_wvf_rep), axis=1))[0]

    unique_labels, test_idx = np.unique(test_cosmos_region, return_inverse=True)

    labelling = {l: i for i, l in enumerate(unique_labels)}
    test_index = [labelling[label] for label in test_cosmos_region]
    
    # Get the logits for the test data
    if model == "mlp":
        clf = load_from_existing_model(
            embedding_model=encoder, freeze=freeze, latent_dim=latent_dim, seed=seed
        )
        test_input_data = np.concatenate(
            [test_acg_rep.reshape(-1, 1010) * 10, test_wvf_rep], axis=1
        )
        all_proba_predictions = clf.predict_proba(test_input_data)
        all_predictions = np.argmax(all_proba_predictions, axis=1)

    elif model == "linear":
        clf = load_from_existing_logistic(embedding_model=encoder, seed=seed)
        test_input_data = np.concatenate([test_acg_rep, test_wvf_rep], axis=1)
        all_proba_predictions = clf.predict_proba(np.nan_to_num(test_input_data))
        all_predictions = np.argmax(all_proba_predictions, axis=1)

    try:
        del test_wvf_rep
        del test_acg_rep
    except:
        pass

    all_predictions[nan_idx] = -1
    all_proba_predictions[nan_idx, :] = -1

    if per_depth:
        all_predictions = all_predictions.reshape(N_test, 5)
        all_proba_predictions = all_proba_predictions.reshape(N_test, 5, -1)
    else:
        all_predictions = all_predictions.reshape(N_test, 1)
        all_proba_predictions = all_proba_predictions.reshape(N_test, 1, -1)

    # Calculate the majority vote
    final_predictions = majority_vote(all_predictions)

    # Calculate the average logits
    final_proba_predictions = average_logits(all_proba_predictions)

    non_nan_idx = np.where(~np.isnan(final_predictions))[0]
    final_predictions = final_predictions[non_nan_idx]
    final_proba_predictions = final_proba_predictions[non_nan_idx]

    if return_depth:
        depth =  depth[non_nan_idx]
        pids = pids[non_nan_idx]
        return (
            final_predictions,
            final_proba_predictions,
            np.array(test_index)[non_nan_idx],
            labelling,
            depth,
            pids,
        )
    else:
        return (
            final_predictions,
            final_proba_predictions,
            np.array(test_index)[non_nan_idx],
            labelling,
        )


# Function to calculate majority vote considering NaNs
def majority_vote(predictions):
    mode_result = []
    for prediction in predictions:
        non_nan_pred = prediction[prediction != -1]
        if non_nan_pred.size > 0:
            values, counts = np.unique(non_nan_pred, return_counts=True)
            mode_result.append(values[np.argmax(counts)])
        else:
            mode_result.append(np.nan)
    return np.array(mode_result)


# Function to calculate average logits considering NaNs
def average_logits(prob_predictions):
    mean_result = []
    for i in range(prob_predictions.shape[0]):
        proba_set = prob_predictions[
            i, :, :
        ]  # Transpose to iterate over each set of logits
        valid_logits = proba_set[proba_set != -1].reshape(-1, prob_predictions.shape[2])
        if valid_logits.size > 0:
            mean_result.append(np.mean(valid_logits, axis=0))
        else:
            mean_result.append(-np.ones(prob_predictions.shape[2]))
    return np.stack(mean_result)


if __name__ == "__main__":
    save_dir = (
        "/projects/bcxj/yxu13/ibl_mwvf/cell-type_representation/ibl_k_nearest_neighbors"
    )
    runrunseed=42
    torch.manual_seed(runrunseed)
    torch.cuda.manual_seed(runrunseed)
    np.random.seed(runrunseed)

    simclr_seed = [42, 26, 29, 65, 70]

    for embedding_model in ["contrastive"]:  # ["contrastive", "vae", "supervise"]:
        for i in range(5):
            if embedding_model == "vae":
                seed = i + 1234
            elif embedding_model == "simclr":
                seed = simclr_seed[i]
            else:
                seed = i + 42
            for model in ["linear"]:  # ["linear", "mlp"]:
                if model == "linear":
                    freeze_list = [True]  # doesn't matter for linear
                elif embedding_model == "supervise":
                    freeze_list = [False]
                else:
                    freeze_list = [True, False]
                for freeze in freeze_list:
                    (
                        final_predictions,
                        final_proba_predictions,
                        test_index,
                        labelling,
                    ) = nearest_neighbor_logits_combine(
                        model=model,
                        latent_dim=512,
                        encoder=embedding_model,
                        per_depth=True,
                        freeze=freeze,
                        seed=seed,
                    )
                    
                    overall_accuracy = balanced_accuracy_score(
                        test_index, final_predictions
                    )
                    overall_f1 = f1_score(
                        test_index, final_predictions, average="macro"
                    )
                    overall_cm = confusion_matrix(
                        test_index, final_predictions, normalize="true"
                    )
                    overall_report = classification_report(
                        test_index,
                        final_predictions,
                        target_names=labelling.keys(),
                        output_dict=True,
                    )
                    final_max_proba_predictions = np.argmax(
                        final_proba_predictions, axis=1
                    )
                    max_proba_accuracy = balanced_accuracy_score(
                        test_index, final_max_proba_predictions
                    )
                    max_proba_f1 = f1_score(
                        test_index, final_max_proba_predictions, average="macro"
                    )
                    max_proba_cm = confusion_matrix(
                        test_index, final_max_proba_predictions, normalize="true"
                    )
                    max_proba_report = classification_report(
                        test_index,
                        final_max_proba_predictions,
                        target_names=labelling.keys(),
                        output_dict=True,
                    )
                    
                    numeric_metrcs = {
                        "test_index": test_index,
                        "majority_vote_accuracy": overall_accuracy,
                        "majority_vote_f1": overall_f1,
                        "majority_vote_cm": overall_cm,
                        "majority_vote_predictions": final_predictions,
                        "max_proba_accuracy": max_proba_accuracy,
                        "max_proba_f1": max_proba_f1,
                        "max_proba_cm": max_proba_cm,
                        "max_proba_predictions":final_max_proba_predictions
                    }
                    save_path = f"{save_dir}/{embedding_model}_freeze{freeze}_seed_{seed}_{model}.pkl"
                    with open(save_path, "wb") as save_:
                        pickle.dump(numeric_metrcs, save_)

                    report_df = pd.DataFrame.from_dict(overall_report)
                    report_df.to_csv(
                        f"{save_dir}/{embedding_model}_freeze{freeze}_seed_{seed}_{model}_majority_vote_report.csv"
                    )

                    max_proba_report_df = pd.DataFrame.from_dict(max_proba_report)
                    max_proba_report_df.to_csv(
                        f"{save_dir}/{embedding_model}_freeze{freeze}_seed_{seed}_{model}_max_proba_report.csv"
                    )

                    gc.collect()
                    torch.cuda.empty_cache()
                    print(
                        f"Finished {embedding_model} with freeze {freeze} with model {model} and seed {seed}"
                    )
                    print(f"majority_vote_accuracy:{overall_accuracy}")
                    print(f"max_proba_accuracy:{max_proba_accuracy}")
                    print(f"majority_vote_f1:{overall_f1}")
                    print(f"max_proba_f1:{max_proba_f1}")
