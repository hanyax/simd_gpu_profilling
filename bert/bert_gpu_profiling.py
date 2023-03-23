import os
import onnx
import onnxruntime
import numpy as np
import json
import collections
import data_processing as dp
import tokenization
from pathlib import Path
import subprocess
import time
from onnxruntime.quantization import CalibrationDataReader, create_calibrator, CalibrationMethod, write_calibration_table, QuantType, QuantizationMode, QDQQuantizer

class BertDataReader(CalibrationDataReader):
    def __init__(self,
                 model_path,
                 squad_json,
                 vocab_file,
                 batch_size,
                 max_seq_length,
                 doc_stride,
                 start_index=0,
                 end_index=0):
        self.model_path = model_path
        self.data = dp.read_squad_json(squad_json)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.example_stride = batch_size # number of examples as one example stride. (set to equal to batch size) 
        self.start_index = start_index # squad example index to start with
        self.end_index = len(self.data) if end_index == 0 else end_index 
        self.current_example_index = start_index
        self.current_feature_index = 0 # global feature index (one example can have more than one feature) 
        self.tokenizer = tokenization.BertTokenizer(vocab_file=vocab_file, do_lower_case=True)
        self.doc_stride = doc_stride 
        self.max_query_length = 64
        self.enum_data_dicts = iter([])
        self.features_list = []
        self.token_list = []
        self.example_id_list = []
        self.start_of_new_stride = False # flag to inform that it's a start of new example stride

    def get_next(self):
        iter_data = next(self.enum_data_dicts, None)
        if iter_data:
            self.start_of_new_stride= False
            return iter_data

        self.enum_data_dicts = None
        if self.current_example_index >= self.end_index:
            print("Reading dataset is done. Total examples is {:}".format(self.end_index-self.start_index))
            return None
        elif self.current_example_index + self.example_stride > self.end_index:
            self.example_stride = self.end_index - self.current_example_index

        if self.current_example_index % 10 == 0:
            current_batch = int(self.current_feature_index / self.batch_size) 
            print("Reading example index {:}, batch {:}, containing {:} sentences".format(self.current_example_index, current_batch, self.batch_size))

        # example could have more than one feature
        # we collect all the features of examples and process them in one example stride
        features_in_current_stride = []
        for i in range(self.example_stride):
            example = self.data[self.current_example_index+ i]
            features = dp.convert_example_to_features(example.doc_tokens, example.question_text, self.tokenizer, self.max_seq_length, self.doc_stride, self.max_query_length)
            self.example_id_list.append(example.id)
            self.features_list.append(features)
            self.token_list.append(example.doc_tokens)
            features_in_current_stride += features
        self.current_example_index += self.example_stride
        self.current_feature_index+= len(features_in_current_stride)


        # following layout shows three examples as example stride with batch size 2:
        # 
        # start of new example stride 
        # |
        # |
        # v
        # <--------------------- batch size 2 ---------------------->
        # |...example n, feature 1....||...example n, feature 2.....| 
        # |...example n, feature 3....||...example n+1, feature 1...| 
        # |...example n+1, feature 2..||...example n+1, feature 3...|
        # |...example n+1, feature 4..||...example n+2, feature 1...|

        data = []
        for feature_idx in range(0, len(features_in_current_stride), self.batch_size):
            input_ids = np.random.randint(0,100, (1,256), dtype='int64')
            input_mask = np.random.randint(0,100, (1,256), dtype='int64')
            segment_ids = np.random.randint(0,100, (1,256), dtype='int64')
            unique_ids = [256]

            # for i in range(self.batch_size):
            #     if feature_idx + i >= len(features_in_current_stride):
            #         break
            #     feature = features_in_current_stride[feature_idx + i]
            #     if len(input_ids) and len(segment_ids) and len(input_mask):
            #         input_ids = np.vstack([input_ids, feature.input_ids])
            #         input_mask = np.vstack([input_mask, feature.input_mask])
            #         segment_ids = np.vstack([segment_ids, feature.segment_ids])
            #     else:
            #         input_ids = np.expand_dims(feature.input_ids, axis=0)
            #         input_mask = np.expand_dims(feature.input_mask, axis=0)
            #         segment_ids = np.expand_dims(feature.segment_ids, axis=0)

            # data.append({"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids":segment_ids, "outypyus":input_mask})
            data.append({"input_ids:0": input_ids, "input_mask:0": input_mask, "segment_ids:0":segment_ids, "unique_ids_raw_output___9:0": unique_ids})

        self.enum_data_dicts = iter(data)
        self.start_of_new_stride = True
        return next(self.enum_data_dicts, None)

def get_predictions(example_id_in_current_stride,
                    features_in_current_stride,
                    token_list_in_current_stride,
                    batch_size,
                    outputs,
                    _NetworkOutput,
                    all_predictions):
                    
    if example_id_in_current_stride == []:
        return 

    base_feature_idx = 0
    for idx, id in enumerate(example_id_in_current_stride):
        features = features_in_current_stride[idx]
        doc_tokens = token_list_in_current_stride[idx]
        networkOutputs = []
        for i in range(len(features)):
            x = (base_feature_idx + i) // batch_size
            y = (base_feature_idx + i) % batch_size

            output = outputs[x]
            start_logits = output[0][y]
            end_logits = output[1][y]

            networkOutputs.append(_NetworkOutput(
                start_logits = start_logits,
                end_logits = end_logits,
                feature_index = i 
                ))

        base_feature_idx += len(features) 

        # Total number of n-best predictions to generate in the nbest_predictions.json output file
        n_best_size = 20

        # The maximum length of an answer that can be generated. This is needed
        # because the start and end predictions are not conditioned on one another
        max_answer_length = 30

        prediction, nbest_json, scores_diff_json = dp.get_predictions(doc_tokens, features,
                networkOutputs, n_best_size, max_answer_length)

        all_predictions[id] = prediction

def inference(data_reader, ort_session):

    _NetworkOutput = collections.namedtuple(  # pylint: disable=invalid-name
            "NetworkOutput",
            ["start_logits", "end_logits", "feature_index"])
    all_predictions = collections.OrderedDict()
    
    example_id_in_current_stride = [] 
    features_in_current_stride = []  
    token_list_in_current_stride = []
    outputs = []
    #while True:
    inputs = data_reader.get_next()

    start = time.time()
    output = ort_session.run(["unstack:0","unstack:1", "unique_ids:0"], inputs)
    end = time.time()
    prof_file = ort_session.end_profiling()
    
    inference_time = np.round((end - start) * 1000, 2)
    print('========================================')
    print('Inference time: ' + str(inference_time) + " ms")
    print('========================================')
    print("Inference Complete")

    return all_predictions

def get_op_nodes_not_followed_by_specific_op(model, op1, op2):
    op1_nodes = []
    op2_nodes = []
    selected_op1_nodes = []
    not_selected_op1_nodes = []

    for node in model.graph.node:
        if node.op_type == op1:
            op1_nodes.append(node)
        if node.op_type == op2:
            op2_nodes.append(node)

    for op1_node in op1_nodes:
        for op2_node in op2_nodes:
            if op1_node.output == op2_node.input:
                selected_op1_nodes.append(op1_node.name)
        if op1_node.name not in selected_op1_nodes:
            not_selected_op1_nodes.append(op1_node.name)

    return not_selected_op1_nodes

if __name__ == '__main__':

    # Model, dataset and quantization settings
    squad_json = "./squad/dev-v1.1.json"
    vocab_file = "./squad/vocab.txt"
    qdq_model_path = "./bert-int8.onnx"
    sequence_lengths = [384, 128] # if use sequence length 384 then choose doc stride 128. if use sequence length 128 then choose doc stride 32.   
    doc_stride = [128, 32]
    calib_num = 100
    op_types_to_quantize = ['MatMul', 'Add']
    batch_size = 1

    # QDQ model inference and get SQUAD prediction 
    os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"  # Enable TRT FP16 precision
    os.environ["ORT_TENSORRT_INT8_ENABLE"] = "1"  # Enable TRT INT8 precision
    batch_size = 1 
    data_reader = BertDataReader(qdq_model_path, squad_json, vocab_file, batch_size, sequence_lengths[-1], doc_stride[-1])
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_options.enable_profiling = True
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = onnxruntime.InferenceSession(qdq_model_path, sess_options=sess_options, providers=providers)
    all_predictions = inference(data_reader, session) 
