import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Argument for catastrophic training.')
    parser.add_argument('--task', action='append', help="Task to be added to model, put each task seperately, Allowed tasks currently are : \nsst \ncola \ntrec \nsubjectivity. If train and evaluate options are not provide they default to tasks option.\n")
    parser.add_argument('--train', action='append', help="Task to train on, put each task seperately, Allowed tasks currently are : \nsst \ncola \ntrec \nsubjectivity\n")
    parser.add_argument('--evaluate', action='append', help="Task to evaluate on, put each task seperately, Allowed tasks currently are : \nsst \ncola \ntrec \nsubjectivity\n")
    parser.add_argument('--few_shot', action='store_true', help="Train task on few shot learning before evaluating.")
    parser.add_argument('--mean_classifier', action='store_true', help="Start using mean classifier instead of normal evaluation.")
    parser.add_argument('--joint', action='store_true', help="Do the joint training or by the task sequentially")
    parser.add_argument('--diff_class', action='store_true', help="Do training with Different classifier for each task")
#    parser.add_argument('--task_diagnostics', action='store_true', help="Enable task diagnostics")

    # CNN Params
    parser.add_argument('--cnn', action='store_true', help="Use CNN")
    parser.add_argument('--mlp', action='store_true', help="Use Multi Layer Preceptron")
    parser.add_argument('--lstm', action='store_true', help="Use LSTM architecture")
    parser.add_argument('--pyramid', action='store_true', help="Use Deep Pyramid CNN works only when --cnn is applied")
    parser.add_argument('--ngram_filter', type=int, default=2, help="Ngram filter size to send in")
    parser.add_argument('--stride', type=int, default=1, help="Strides to use for CNN")


    parser.add_argument('--require_empty', action="store_true", help="Require the folder to be empty")
    parser.add_argument('--epochs', type=int, default=1000, help="Number of epochs to train for")
    parser.add_argument('--layers', type=int, default=1, help="Number of layers")
    parser.add_argument('--dropout', type=float, default=0, help="Use dropout")
    parser.add_argument('--batch_norm', action='store_true', help="Use Batch Normalisation Only MLP for now")
    parser.add_argument('--bs', type=int, default=64, help="Batch size to use")
    parser.add_argument('--bidirectional', action='store_true', help="Run LSTM Network using bi-directional network.")
    parser.add_argument('--embeddings', help="Use which embedding ElMO embeddings or BERT",type=str, default='default')

    # Optimization Based Parameters
    parser.add_argument('--wdecay', type=float, help="L2 Norm to use")
    parser.add_argument('--lr', type=float, default=0.001, help="L2 Norm to use")
    parser.add_argument('--opt_alg', type=str, default="adam", help="Optimization algorithm to use")
    parser.add_argument('--patience', type=int, default=10, help="Number of layers")


    parser.add_argument('--e_dim', type=int, default=128, help="Embedding Dimension")
    parser.add_argument('--h_dim', type=int, default=1150, help="Hidden Dimension")
    parser.add_argument('--s_dir', help="Serialization directory")
    parser.add_argument('--transformer', help="Use transformer unit",action='store_true')
#    parser.add_argument('--transposed', help="Use the transposed with sequence first in transformer",action='store_true')
    parser.add_argument('--train_embeddings', help="Enable fine-tunning of embeddings like elmo",action='store_true')
    parser.add_argument('--IDA', help="Use IDA Encoder",action='store_true')
    parser.add_argument('--hashed', help="Use Hashed Memory Networks",action='store_true')
    parser.add_argument('--ewc', help="Use Elastic Weight consolidation",action='store_true')
    parser.add_argument('--oewc', help="Use Our during training fisher Elastic Weight consolidation",action='store_true')
    parser.add_argument('--ewc_importance', type=int, default=1000, help="Use Elastic Weight consolidation importance to add weights")
    parser.add_argument('--ewc_samples', type=int, default=240, help="Number of samples to use for training ewc loss")
    parser.add_argument('--ewc_normalise', type=str, help="Use Elastic Weight consolidation length, batches, none")

    ## options to embed position and task information
    parser.add_argument('--task_embed', action='store_true', help="Use the task encoding to encode task id")
    parser.add_argument('--task_encode', action='store_true', help="Use the task encoding to encode task id using transformer style position encoding")
    parser.add_argument('--position_embed', action='store_true', help="Add the positional embeddings in the word embeddings.")
    parser.add_argument('--no_positional', help="Disable positional embeddings in transformer",action='store_true')

    ## Memory related options
    parser.add_argument('--mem_size', help="Memory key size", type=int, default=300)
    parser.add_argument('--mem_context_size', help="Memory output size", type=int, default=512)
    parser.add_argument('--use_memory', action='store_true', help="Weather to use memory are not")
    parser.add_argument('--use_task_memory', action='store_true', help="Weather to use task memory before final classification layer")
    parser.add_argument('--use_binary', action='store_true', help="Make the memory access binary")
    parser.add_argument('--pad_memory', action='store_true', help="Pad the Memory after training each task")


    parser.add_argument('--inv_temp', help="Inverse temp to use for IDA or other algorithms",type=float, default=None)
    parser.add_argument('--temp_inc', help="Increment in temperature after each task",type=float, default=None)
    parser.add_argument('--softmax_temp', help="Increment in temperature after the softmax activation", action='store_true')
    parser.add_argument('--all_temp', help="Increment in temperature after the softmax activation", action='store_true')
    parser.add_argument('--emb_temp', help="Increment in temperature after the embedding activation", action='store_true')
    parser.add_argument('--enc_temp', help="Increment in temperature after the Encoder activation", action='store_true')
    parser.add_argument('--layer_temp', help="Increment in temperature after each layer", action='store_true')

    parser.add_argument('--majority', help="Use Sequence to sequence",action='store_true')
    parser.add_argument('--tryno', type=int, default=1, help="This is ith try add this to name of df")
    parser.add_argument('--small', type=int, default=None, help="Use only these examples from each set")
    parser.add_argument('--run_name', type=str, default="Default", help="This is the run name being saved to tensorboard")
    parser.add_argument('--storage_prefix', type=str, default="./runs/", help="This is used to store the runs inside runs folder")

    parser.add_argument('--pooling', type=str, default="max", help="Selects the pooling operation for CNN, max pooling, min pooling, average pooling. max,min,avg")
    parser.add_argument('--no_save_weight', action='store_true', help="Disable saving of weights")

    args = parser.parse_args()
    if args.task_embed and args.transformer:
        if args.e_dim % 2 == 0:
            print("Need odd dimension for task embedding and transformer, reducing by one to make odd")
            args.e_dim -= 1
    if (args.transformer or args.cnn or args.lstm) and args.batch_norm:
        raise Exception("Batch Normalisation is currently only supported for MLP")
    return args

