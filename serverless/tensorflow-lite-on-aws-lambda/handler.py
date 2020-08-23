import json
import numpy as np

import tflite_runtime.interpreter as tflite


def predict(event, context):
    # retrieve entry event -> json_list_list_x
    json_list_list_x = event.get('body')
    # test
    #print(json_list_list_x[0])

    list_list_in = json.loads(json.loads(json_list_list_x))
    # convert in list of array
    # retrieve json into lambda
    list_arr_in = []
    for list_curr in list_list_in:
        list_arr_in.append(np.array(list_curr))

    print("INPUT : nb. arrays : {} / arrays shape: {}".format(len(list_arr_in), 
                                             list_arr_in[0].shape))

    # prepare TFlite model
    interpreter = tflite.Interpreter(model_path='converted_model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


    # Run the model with TensorFlow Lite
    list_list_out = []
    for x_multi in list_arr_in:
        interpreter.set_tensor(input_details[0]["index"], 
                               x_multi.astype(np.float32))
        interpreter.invoke()
        result = interpreter.get_tensor(output_details[0]["index"])
        list_list_out.append(result.tolist())
        # Please note: TfLite fused Lstm kernel is stateful, so we need to reset
        # the states.
        # Clean up internal states.
        interpreter.reset_all_variables()
    print("OUTPUT : nb. arrays : {} / arrays shape in list: {}" \
          .format(len(list_list_out), np.array(list_list_out[0]).shape))

    # Prepare output
    json_list_list_out = json.dumps(list_list_out)
    response = {
        "statusCode": 200,
        "body": json_list_list_out
    }
    return response