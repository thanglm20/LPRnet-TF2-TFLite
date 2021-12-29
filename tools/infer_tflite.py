import sys
import numpy as np
import tensorflow as tf
from PIL import Image
from argparse import ArgumentParser

DICT_OCR = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z']


def build_argparser():
  parser = ArgumentParser()
  parser.add_argument('--model', help='Path to tflite file with a trained model.', required=True, type=str)
  parser.add_argument('--image', help='Image with license plate', required=True, type=str)
  return parser

def main():

    args = build_argparser().parse_args()

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img_in = Image.open(args.image)
    img_in.show()
    img_in = img_in.convert('RGB')
    img_in = img_in.resize((input_details[0]['shape'][2], input_details[0]['shape'][1]))
    input_array = np.array(img_in , dtype=np.float32)
    input_array = input_array / 255.0
    input_array = np.reshape(input_array, input_details[0]['shape'])
    interpreter.set_tensor(input_details[0]['index'], input_array)
    interpreter.invoke()

    tens_out = {'input': input_array}
    for i in range(len(output_details)):
        tens_out.update({output_details[i]['name']: interpreter.get_tensor(output_details[i]['index'])})

    out = interpreter.get_tensor(output_details[0]['index'])
    out_char_codes = [np.argmax(out[0][i]) for i in range(out.shape[1])]

    prob = tf.transpose(out, (1, 0, 2))  # prepare for CTC
    data_length = tf.fill([tf.shape(prob)[1]], tf.shape(prob)[0])  # input seq length, batch size
    graph_output = tf.placeholder(tf.float32, shape=[None, 1, 2]) # graph_output has a dynamic shape
    tf_greedy_path, _ = tf.nn.ctc_greedy_decoder(prob, data_length, merge_repeated=True)
    tf_greedy_path = tf.convert_to_tensor([tf.sparse_tensor_to_dense(sparse_tensor) for sparse_tensor in tf_greedy_path])

    list_out = []
    with tf.Session() as sess:
        list_out = sess.run(tf_greedy_path, feed_dict={graph_output:[[[1., 2.]], [[3., 4.]]]})

    output = np.array(list_out[0][0])
    text_out = []
    for i in output:
        text_out += DICT_OCR[i]

            
    print("****************************************************************")
    print("Result: ", text_out)
    print("****************************************************************")


if __name__ == "__main__":
    main()