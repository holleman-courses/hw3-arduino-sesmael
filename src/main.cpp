#include <Arduino.h>
#include <TensorFlowLite.h>
#include "sin_predictor_model.h"  // sin predictor model
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/version.h>

#define INPUT_BUFFER_SIZE 64
#define OUTPUT_BUFFER_SIZE 64
#define INT_ARRAY_SIZE 8  
#define EXPECTED_INPUT_SIZE 7  

//tensorFlow lite components
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;
constexpr int tensor_arena_size = 2 * 1024;
uint8_t tensor_arena[tensor_arena_size];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// Input buffers
char received_char = (char)NULL;
int chars_avail = 0;
char out_str_buff[OUTPUT_BUFFER_SIZE];
char in_str_buff[INPUT_BUFFER_SIZE];
int input_array[INT_ARRAY_SIZE];
int in_buff_idx = 0;
int array_length = 0;

// Function declarations
int string_to_array(char *in_str, int *int_array);
void print_int_array(int *int_array, int array_len);
int sum_array(int *int_array, int array_len);

void setup() {
    delay(5000);
    Serial.begin(115200);
    Serial.println("Initializing TensorFlow Lite model...");

    //Loading the model
    model = tflite::GetModel(sin_predictor_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema version mismatch!");
        return;
    }

    // Initialize interpreter
    interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, tensor_arena_size, &micro_error_reporter);
    interpreter->AllocateTensors();
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    Serial.println("TFLM Model Initialized.");
}

void loop() {
    chars_avail = Serial.available();
    if (chars_avail > 0) {
        received_char = Serial.read();
        Serial.print(received_char); 
        in_str_buff[in_buff_idx++] = received_char;

        if (received_char == 13) { // user pressed Enter
            Serial.println("\nProcessing input...");

            // Convert input string to an integer array
            array_length = string_to_array(in_str_buff, input_array);

            if (array_length != EXPECTED_INPUT_SIZE) {
                Serial.println("Error: Please enter exactly 7 numbers");
            } else {
                
                Serial.print("Input Array: ");
                print_int_array(input_array, array_length);

                //Measure time for printing
                unsigned long t0 = micros();
                Serial.println("Timing Test Statement");
                unsigned long t1 = micros();

                //scaling and zero-point for input quantization
                float input_scale = input_tensor->params.scale;
                int input_zero_point = input_tensor->params.zero_point;

                //load quantized input into TensorFlow Lite model
                for (int i = 0; i < EXPECTED_INPUT_SIZE; i++) {
                    int8_t quantized_value = static_cast<int8_t>((input_array[i] / input_scale) + input_zero_point);
                    input_tensor->data.int8[i] = quantized_value;
                }

                // Measure inference time
                unsigned long t2 = micros();
                interpreter->Invoke(); // Run inference
                unsigned long t3 = micros();

                //Retrieve model output scaling parameters
                float output_scale = output_tensor->params.scale;
                int output_zero_point = output_tensor->params.zero_point;
                int8_t raw_prediction = output_tensor->data.int8[0];

                //Correctly dequantize output
                int prediction = static_cast<int>((raw_prediction - output_zero_point) * output_scale);

                //Print results
                Serial.print("Model Prediction: ");
                Serial.println(prediction);

                //Measure execution time
                unsigned long t_print = t1 - t0;
                unsigned long t_infer = t3 - t2;
                Serial.print("Print Time (µs): ");
                Serial.println(t_print);
                Serial.print("Inference Time (µs): ");
                Serial.println(t_infer);
            }

            // Clear the input buffer
            memset(in_str_buff, (char)0, INPUT_BUFFER_SIZE * sizeof(char));
            in_buff_idx = 0;
        } else if (in_buff_idx >= INPUT_BUFFER_SIZE) {
            memset(in_str_buff, (char)0, INPUT_BUFFER_SIZE * sizeof(char));
            in_buff_idx = 0;
        }
    }
}

int string_to_array(char *in_str, int *int_array) {
    int num_integers = 0;
    char *token = strtok(in_str, ",");

    while (token != NULL) {
        int_array[num_integers++] = atoi(token);
        token = strtok(NULL, ",");
        if (num_integers >= INT_ARRAY_SIZE) {
            break;
        }
    }

    return num_integers;
}

void print_int_array(int *int_array, int array_len) {
    int curr_pos = 0;
    sprintf(out_str_buff, "Integers: [");
    curr_pos = strlen(out_str_buff);

    for (int i = 0; i < array_len; i++) {
        curr_pos += sprintf(out_str_buff + curr_pos, "%d, ", int_array[i]);
    }
    sprintf(out_str_buff + curr_pos, "]\r\n");
    Serial.print(out_str_buff);
}
