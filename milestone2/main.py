import argparse
import numpy as np

from src.methods.dummy_methods import DummyClassifier
from src.methods.mlp import MLP
from src.losses import MSE
from src.activations import Sigmoid, ReLU, Identity
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn, get_n_classes , onehot_to_label , label_to_onehot
import os

np.random.seed(100)


def main(args):
    """
    The main function of the script.

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """


    dataset_path = args.data_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    ## 1. We first load the data.

    feature_data = np.load(dataset_path, allow_pickle=True)
    train_features, test_features, train_labels_reg, test_labels_reg, train_labels_classif, test_labels_classif = (
        feature_data['xtrain'],feature_data['xtest'],feature_data['ytrainreg'],
        feature_data['ytestreg'],feature_data['ytrainclassif'],feature_data['ytestclassif']
    )

    ## 2. Then we must prepare it. This is where you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        num_val = int(0.2 * train_features.shape[0])
        val_features = train_features[:num_val]
        val_labels_reg = train_labels_reg[:num_val]
        val_labels_classif = train_labels_classif[:num_val]
        train_features = train_features[num_val:]
        train_labels_reg = train_labels_reg[num_val:]
        train_labels_classif = train_labels_classif[num_val:]
        test_features = val_features
        test_labels_reg = val_labels_reg
        test_labels_classif = val_labels_classif
        pass

    ### WRITE YOUR CODE HERE to do any other data processing
    mean = np.mean(train_features, axis=0, keepdims=True)
    std = np.std(train_features, axis=0, keepdims=True)
    train_features = normalize_fn(train_features, mean, std)
    test_features = normalize_fn(test_features, mean, std)

    n_features = train_features.shape[1]
    n_classes = get_n_classes(train_labels_classif)

    ## 3. Initialize the method you want to use.

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "kmeans":
        method_obj = KMeans(K=args.K, max_iters=args.max_iters)
        pass

    elif args.method == "mlp":
        if args.task == "classification":
            hidden_dims = [8, 8]
            dimensions = (n_features, *hidden_dims, n_classes)
            activations = [ReLU] * len(hidden_dims) + [Sigmoid]
        elif args.task == "regression":
            hidden_dims = [64, 32]
            dimensions = (n_features, *hidden_dims, 1)
            activations = [ReLU] * len(hidden_dims) + [Identity]
        else:
            raise ValueError(f"Unknown task: {args.task}")
        method_obj = MLP(dimensions=dimensions, activations=activations)
        pass
    else:
        raise ValueError(f"Unknown method: {args.method}")

    ## 4. Train and evaluate the method

    if args.task == "classification":

        if args.method == "mlp":
            # MLP expects one-hot targets; we convert predictions back to class indices.
            y_train_oh = label_to_onehot(train_labels_classif, C=n_classes)
            method_obj.fit(train_features, y_train_oh, loss=MSE,
                           epochs=args.max_iters, batch_size=args.nn_batch_size,
                           learning_rate=args.lr)
            preds_train = onehot_to_label(method_obj.predict(train_features))
            preds = onehot_to_label(method_obj.predict(test_features))
        else:
            # DummyClassifier / KMeans: standard fit/predict on integer labels
            preds_train = method_obj.fit(train_features, train_labels_classif)
            preds = method_obj.predict(test_features)

        acc = accuracy_fn(preds_train, train_labels_classif)
        macrof1 = macrof1_fn(preds_train, train_labels_classif)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(preds, test_labels_classif)
        macrof1 = macrof1_fn(preds, test_labels_classif)
        print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
        pass

    elif args.task == "regression":
        assert args.method != "kmeans", f"You should use kmeans as a classification method"

        ### WRITE YOUR CODE HERE
        assert args.method != "kmeans", f"You should use kmeans as a classification method"

        if args.method == "mlp":
            # MLP expects 2D targets (N, 1); we squeeze predictions back to (N,).
            y_train = train_labels_reg.reshape(-1, 1).astype(float)
            method_obj.fit(train_features, y_train, loss=MSE,
                           epochs=args.max_iters, batch_size=args.nn_batch_size,
                           learning_rate=args.lr)
            preds_train = method_obj.predict(train_features).squeeze()
            preds = method_obj.predict(test_features).squeeze()
        else:
            preds_train = method_obj.fit(train_features, train_labels_reg)
            preds = method_obj.predict(test_features)

        train_mse = mse_fn(preds_train, train_labels_reg)
        print(f"\nTrain set: MSE = {train_mse:.6f}")

        test_mse = mse_fn(preds, test_labels_reg)
        print(f"Test set:  MSE = {test_mse:.6f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="classification",
        type=str,
        help="classification / regression / clustering",
    )
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / kmeans / mlp",
    )
    parser.add_argument(
        "--data_path",
        default="data/features.npz",
        type=str,
        help="path to your dataset CSV file",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=35,
        help="number of clusters datapoints used for kmeans",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iterations (KMeans) / epochs (MLP)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, "
             "otherwise use a validation set",
    )
    # Feel free to add more arguments here if you need!
    parser.add_argument(
        "--nn_batch_size",
        type=int,
        default=64,
        help="batch size for the MLP",
    )
    args = parser.parse_args()
    main(args)
