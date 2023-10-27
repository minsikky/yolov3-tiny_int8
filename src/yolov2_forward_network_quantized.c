#include "additionally.h"    // some definitions from: im2col.h, blas.h, list.h, utils.h, activations.h, tree.h, layer.h, network.h
// softmax_layer.h, reorg_layer.h, route_layer.h, region_layer.h, maxpool_layer.h, convolutional_layer.h

#define GEMMCONV

//#define SSE41
//#undef AVX

#define MAX_VAL_8 127    // 7-bit (1-bit sign)
#define MAX_VAL_16 32767    // 15-bit (1-bit sign)
#define MAX_VAL_32 2147483647 // 31-bit (1-bit sign)
#define R_MULT 1

int max_abs(int src, int max_val)
{
    if (abs(src) > abs(max_val)) src = (src > 0) ? max_val : -max_val;
    return src;
}

short int max_abs_short(short int src, short int max_val)
{
    if (abs(src) > abs(max_val)) src = (src > 0) ? max_val : -max_val;
    return src;
}

// im2col.c
int8_t im2col_get_pixel_int8(int8_t *im, int height, int width, int channels,
    int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

// im2col.c
//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu_int8(int8_t* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, int8_t* data_col)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel_int8(data_im, height, width, channels,
                    im_row, im_col, c_im, pad);
            }
        }
    }
}

void gemm_nn_int8_int16(int M, int N, int K, int8_t ALPHA,
    int8_t *A, int lda,
    int8_t *B, int ldb,
    int16_t *C, int ldc)
{
    int32_t *c_tmp = calloc(N, sizeof(int32_t));
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            register int16_t A_PART = ALPHA*A[i*lda + k];
            //#pragma simd parallel for
            for (j = 0; j < N; ++j) {
                c_tmp[j] += A_PART*B[k*ldb + j];
            }
        }
        for (j = 0; j < N; ++j) {
            C[i*ldc + j] += max_abs(c_tmp[j] / R_MULT, MAX_VAL_16);
            c_tmp[j] = 0;
        }
    }
    free(c_tmp);
}

void gemm_nn_int8_int32(int M, int N, int K, int8_t ALPHA,
    int8_t *A, int lda,
    int8_t *B, int ldb,
    int32_t *C, int ldc)
{
    int32_t *c_tmp = calloc(N, sizeof(int32_t));
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            register int16_t A_PART = ALPHA*A[i*lda + k];
            //#pragma simd parallel for
            for (j = 0; j < N; ++j) {
                c_tmp[j] += A_PART*B[k*ldb + j];
            }
        }
        for (j = 0; j < N; ++j) {
            //C[i*ldc + j] += max_abs(c_tmp[j] / R_MULT, MAX_VAL_32);
            C[i*ldc + j] += max_abs(c_tmp[j], MAX_VAL_32);
            c_tmp[j] = 0;
        }
    }
    free(c_tmp);
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            C[i*ldc + j] = truncf(C[i*ldc + j] / R_MULT);
        }
    }
}

void forward_convolutional_layer_q(network net, layer l, network_state state, int save_interim_result)
{

    int out_h = (l.h + 2 * l.pad - l.size) / l.stride + 1;    // output_height=input_height for stride=1 and pad=1
    int out_w = (l.w + 2 * l.pad - l.size) / l.stride + 1;    // output_width=input_width for stride=1 and pad=1
    int i, j;
    int const out_size = out_h*out_w;

    typedef int32_t conv_t;    // l.output
    conv_t *output_q = calloc(l.outputs, sizeof(conv_t));

    
    //For saving interim results
    char input[50];
    char convolved[50];
    char biased[50];
    char activated[50];
    char quantized[50];
    
    if (save_interim_result == 1) {
        snprintf(input, sizeof(input), "interim_result/%d_CONV_0_input.txt", state.index);
        snprintf(convolved, sizeof(convolved), "interim_result/%d_CONV_1_convolved.txt", state.index);
        snprintf(biased, sizeof(biased), "interim_result/%d_CONV_2_biased.txt", state.index);
        snprintf(activated, sizeof(activated), "interim_result/%d_CONV_3_activated.txt", state.index);
        snprintf(quantized, sizeof(quantized), "interim_result/%d_CONV_4_quantized.txt", state.index);
    }

    int x;
    int y;
    int z;
    int w;

    // Input quantization; essentially does the same job as activation quantization
    state.input_int8 = (int8_t *)calloc(l.inputs, sizeof(int));

    for (z = 0; z < l.inputs; ++z) {
        int32_t src = state.input[z] * l.input_quant_multiplier;
        state.input_int8[z] = max_abs(src, MAX_VAL_8);
    }

    if (save_interim_result == 1) {
        FILE *fp_i = fopen(input, "w");
        for (x = 0; x < l.h*l.w; x = x + 1) {
            for (y = 0; y < l.c; y = y + 1) {
                int idx = x + y*l.h*l.w;
                uint8_t first = state.input_int8[idx];
                fprintf(fp_i, "%02x\n", first);
            }
        }
        fclose(fp_i);
    }

    // Convolution
    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = out_h*out_w;
    int8_t *a = l.weights_int8;
    int8_t *b = (int8_t *)state.workspace;
    conv_t *c = output_q;    // int32_t

    // Use GEMM (as part of BLAS)
    im2col_cpu_int8(state.input_int8, l.c, l.h, l.w, l.size, l.stride, l.pad, b);
    int t;    // multi-thread gemm
    #pragma omp parallel for
    for (t = 0; t < m; ++t) {
        gemm_nn_int8_int32(1, n, k, 1, a + t*k, k, b, n, c + t*n, n);
    }
    free(state.input_int8);
    if (save_interim_result == 1) {
        FILE *fp_c = fopen(convolved, "w");

        // Assume Output Order: [channel, width, height]
        for (x = 0; x < out_size; x = x + 1) {
            for (y = 0; y < l.n; y = y + 1) {
                int idx = x + y*out_size;
                uint16_t first = output_q[idx];
                fprintf(fp_c, "%04x\n", first);
            }
        }
        fclose(fp_c);
    }

    // Bias addition
    int fil;
    for (fil = 0; fil < l.n; ++fil) {
        for (j = 0; j < out_size; ++j) {
            output_q[fil*out_size + j] = output_q[fil*out_size + j] + l.biases_quant[fil];
        }
    }

    if (save_interim_result == 1) {
        FILE *fp_b = fopen(biased, "w");     
        for (x = 0; x < out_size; x = x + 1) {
            for (y = 0; y < l.n; y = y + 1) {
                int idx = x + y*out_size;
                uint16_t first = output_q[idx];
                fprintf(fp_b, "%04x\n", first);
            }
        }
        fclose(fp_b);
    }
    
    // Activation
    if (l.activation == LEAKY) {
        for (i = 0; i < l.n*out_size; ++i) {
            output_q[i] = (output_q[i] > 0) ? output_q[i] : output_q[i] / 8; // (?) Changing the divisor from 10 to 8 actually increases the mAP!
        }
    }

    if (save_interim_result == 1) {
        FILE *fp_a = fopen(activated, "w");     
        for (x = 0; x < out_size; x = x + 1) {
            for (y = 0; y < l.n; y = y + 1) {
                int idx = x + y*out_size;
                uint16_t first = output_q[idx];
                fprintf(fp_a, "%04x\n", first);
            }
        }
        fclose(fp_a);
    }

    // De-scaling
    float ALPHA1 = R_MULT / (l.input_quant_multiplier * l.weights_quant_multiplier);
    for (i = 0; i < l.outputs; ++i) {
        l.output[i] = output_q[i] * ALPHA1;
    }

    if (save_interim_result == 1) {
        int z;
        int next_input_quant_multiplier = 1;
        for (z = state.index+1; z < net.n; ++z) {
            if (net.layers[z].type == CONVOLUTIONAL) {
                next_input_quant_multiplier = net.layers[z].input_quant_multiplier;
                break;
            }
        }
        FILE *fp_q = fopen(quantized, "w");   
        for (x = 0; x < out_size; x = x + 1) {
            for (y = 0; y < l.n; y = y + 1) {
                int idx = x + y*out_size;
                int8_t first = max_abs(output_q[idx] * ALPHA1 * next_input_quant_multiplier, MAX_VAL_8);
                uint8_t first_new = first;
                fprintf(fp_q, "%02x\n", first_new);
            }
        }
        fclose(fp_q);
    }

    free(output_q);
}

void yolov2_forward_network_q(network net, network_state state, int save_interim_result)
{
    state.workspace = net.workspace;
    int i;
    for (i = 0; i < net.n; ++i) {
        state.index = i;
        layer l = net.layers[i];

        if (l.type == CONVOLUTIONAL) {
            forward_convolutional_layer_q(net, l, state, save_interim_result);
        }
        else if (l.type == MAXPOOL) {
            forward_maxpool_layer_cpu(l, state);
        }
        else if (l.type == ROUTE) {
            forward_route_layer_cpu(l, state);
        }
        else if (l.type == REORG) {
            forward_reorg_layer_cpu(l, state);
        }
        else if (l.type == UPSAMPLE) {
            forward_upsample_layer_cpu(l, state);
        }
        else if (l.type == SHORTCUT) {
            forward_shortcut_layer_cpu(l, state);
        }
        else if (l.type == YOLO) {
            forward_yolo_layer_cpu(l, state);
        }
        else if (l.type == REGION) {
            forward_region_layer_cpu(l, state);
        }
        else {
            printf("\n layer: %d \n", l.type);
        }
        state.input = l.output;
    }
}

// detect on CPU
float *network_predict_quantized(network net, float *input, int save_interim_result)
{
    network_state state;
    state.net = net;
    state.index = 0;
    state.input = input;
    state.truth = 0;
    state.train = 0;
    state.delta = 0;

    yolov2_forward_network_q(net, state, save_interim_result);    // network on CPU
                                            //float *out = get_network_output(net);
    int i;
    for (i = net.n - 1; i > 0; --i) if (net.layers[i].type != COST) break;
    return net.layers[i].output;
}

/* Quantization-related */

// Get the distribution in the form of histogram data
int * get_distribution(float *arr_ptr, int arr_size, int number_of_ranges, float start_range)
{
    int *count = calloc(number_of_ranges, sizeof(int));

    int i, j;
    for (i = 0; i < arr_size; ++i) {
        float w = arr_ptr[i];

        float cur_range = start_range;
        for (j = 0; j < number_of_ranges; ++j) {
            if (fabs(cur_range) <= w && w < fabs(cur_range * 2))
                count[j]++;
            cur_range *= 2;
        }
    }
    return count;
}

// // Get the scale factor to be multiplied to the weights using the distribution
// float get_scale(float *arr_ptr, int arr_size, int bits_length)
// {
//     const int number_of_ranges = 32;
//     const float start_range = 1.F / 65536;

//     int i, j;
//     int *count = get_distribution(arr_ptr, arr_size, number_of_ranges, start_range);

//     int max_count_range = 0;
//     int index_max_count = 0;
//     for (j = 0; j < number_of_ranges; ++j) {
//         int counter = 0;
//         for (i = j; i < (j + bits_length) && i < number_of_ranges; ++i) {
//             counter += count[i];
//         }
//         if (max_count_range < counter) {
//             max_count_range = counter;
//             index_max_count = j;
//         }
//     }
//     float multiplier = 1 / (start_range * powf(2., (float)index_max_count));
//     free(count);
//     return multiplier;
// }

// Get the scale factor to be multiplied to the weights using the distribution
float get_scale(float *arr_ptr, int arr_size, int bits_length)
{
    const int number_of_ranges = 32;
    const float start_range = 1.F / 65536;

    int i, j;
    //int *count = get_distribution(arr_ptr, arr_size, number_of_ranges, start_range);

    float MSE_min = FLT_MAX;
    float multiplier = start_range;

    for (j = 0; j < number_of_ranges; ++j) {
        float MSE = 0;
        for (i = 0; i < arr_size; ++i) {
            int8_t quantized = max_abs((int)((*(arr_ptr+i)) * (start_range * powf(2., (float)j))), MAX_VAL_8);
            // if (i == 0) fprintf(stderr, "quantized: %d\n", quantized);
            float dequantized = (float)quantized / (start_range * powf(2., (float)j));
            float error = dequantized - (*(arr_ptr+i));
            MSE += error*error;
        }
        //fprintf(stderr, "Multiplier: %f, MSE: %f\n", start_range * powf(2., (float)j), MSE);
        if (MSE < MSE_min) {
            MSE_min = MSE;
            multiplier = start_range * powf(2., (float)j);
        }
    }
    return multiplier;
}

void do_quantization(network net) {
    int counter = 0;
    printf("Multipler    Input    Weight    Bias\n");
    int j;
    for (j = 0; j < net.n; ++j) {
        layer *l = &net.layers[j];
        if (l->type == CONVOLUTIONAL) { // Quantize conv layer only
            size_t const weights_size = l->size*l->size*l->c*l->n;
            size_t const filter_size = l->size*l->size*l->c;

            int i, fil;

            // Input Scaling
            if (counter >= net.input_calibration_size) {
                printf(" Warning: CONV%d has no corresponding input_calibration parameter - default value 16 will be used;\n", j);
            }
            l->input_quant_multiplier = (counter < net.input_calibration_size) ? net.input_calibration[counter] : 16;  // Using 16 as input_calibration as default value
            ++counter;

            // Weight Quantization
            l->weights_quant_multiplier = get_scale(l->weights, weights_size, 8);
            // if (j == 0) l->weights_quant_multiplier = 8;
            // if (j == 2) l->weights_quant_multiplier = 128;
            // if (j == 4) l->weights_quant_multiplier = 256;
            // if (j == 6) l->weights_quant_multiplier = 512;
            // if (j == 8) l->weights_quant_multiplier = 256;
            // if (j == 10) l->weights_quant_multiplier = 512;
            // if (j == 12) l->weights_quant_multiplier = 256;
            // if (j == 13) l->weights_quant_multiplier = 128;
            // if (j == 16) l->weights_quant_multiplier = 256;
            // if (j == 19) l->weights_quant_multiplier = 256;
            // if (j == 20) l->weights_quant_multiplier = 128;
            for (fil = 0; fil < l->n; ++fil) {
                for (i = 0; i < filter_size; ++i) {
                    float w = l->weights[fil*filter_size + i] * l->weights_quant_multiplier; // Scale
                    l->weights_int8[fil*filter_size + i] = max_abs(w, MAX_VAL_8); // Clip
                }
            }

            // Bias Quantization
            float biases_multiplier = (l->weights_quant_multiplier * l->input_quant_multiplier / R_MULT);
            for (fil = 0; fil < l->n; ++fil) {
                float b = l->biases[fil] * biases_multiplier; // Scale
                l->biases_quant[fil] = max_abs(b, MAX_VAL_32); // Clip
            }
            printf(" CONV%d: \t%g \t%g \t%g \n", j, l->input_quant_multiplier, l->weights_quant_multiplier, biases_multiplier);
            //printf(" CONV%d multipliers: input %g, weights %g, bias %g \n", j, l->input_quant_multiplier, l->weights_quant_multiplier, biases_multiplier);
        }
        else {
            //printf(" No quantization for layer %d (layer type: %d) \n", j, l->type);
        }
    }
}

// Save quantized weights, bias, and scale
void save_quantized_model(network net) {
    int j;
    for (j = 0; j < net.n; ++j) {
        layer *l = &net.layers[j];
        if (l->type == CONVOLUTIONAL) {
            size_t const weights_size = l->size*l->size*l->c*l->n;
            size_t const filter_size = l->size*l->size*l->c;

            printf(" Saving quantized weights, bias, and scale for CONV%d \n", j);

            char weightfile[30];
            char weightfile_reordered[30];
            char biasfile[30];
            char scalefile[30];

            sprintf(weightfile, "weights/CONV%d_W.txt", j);
            sprintf(weightfile_reordered, "weights/CONV%d_W_RO.txt", j);
            sprintf(biasfile, "weights/CONV%d_B.txt", j);
            sprintf(scalefile, "weights/CONV%d_S.txt", j);

            int k;

            int x;
            int y;
            int z;
            int w;

            int o;

            FILE *fp_w_ro = fopen(weightfile_reordered, "w");

            // Assume Weight Order: [outchannel, inchannel, width, height]
            if (j == 0) {
                for (o = 0; o < l->n; o = o + 4) {
                    for (x = 0; x < l->size; x = x + 1) {
                        for (y = 0; y < l->size; y = y + 1) {
                            for (z = 0; z < 4; z = z + 1) {
                                for (w = 0; w < l->c; w = w + 1) {
                                    int idx = x*l->size + y + (z+o)*l->size*l->size*l->c + w*l->size*l->size;
                                    uint8_t first = l->weights_int8[idx];
                                    fprintf(fp_w_ro, "%02x\n", first);
                                }
                            }
                        }
                    }
                }
            }
            else if (j == 13 || j == 16 || j == 20) { // 1x1 convolution
                for (o = 0; o < l->n; o = o + 8) {
                    for (x = 0; x < l->c; x = x + 16) {
                        for (z = 0; z < 8; z = z + 1) {
                            for (w = 0; w < 16; w = w + 1) {
                                int idx = (o+z)*l->size*l->size*l->c + (w+x)*l->size*l->size;
                                uint8_t first;
                                if (idx > l->n*l->c*l->size*l->size - 1) break;
                                else first = l->weights_int8[idx];
                                fprintf(fp_w_ro, "%02x\n", first);
                            }
                        }
                    }
                }
            }
            else {
                for (o = 0; o < l->n; o = o + 1) {
                    for (x = 0; x < l->c; x = x + 16) {
                        for (y = 0; y < l->size; y = y + 1) {
                            for (z = 0; z < l -> size; z = z +1) {
                                for (w = 0; w < 16; w = w + 1) {
                                    int idx = o*l->size*l->size*l->c + y*l->size + z + (w+x)*l->size*l->size;
                                    uint8_t first = l->weights_int8[idx];
                                    fprintf(fp_w_ro, "%02x\n", first);
                                }
                            }
                        }
                    }
                }
            }
            fclose(fp_w_ro);

            FILE *fp_w = fopen(weightfile, "w");
            
            for (k = 0; k < weights_size; k = k + 4) {
                uint8_t first = k < weights_size ? l->weights_int8[k] : 0;
                uint8_t second = k+1 < weights_size ? l->weights_int8[k+1] : 0;
                uint8_t third = k+2 < weights_size ? l->weights_int8[k+2] : 0;
                uint8_t fourth = k+3 < weights_size ? l->weights_int8[k+3] : 0;
                fprintf(fp_w, "%02x%02x%02x%02x\n", first, second, third, fourth);
            }
            
            fclose(fp_w);

            FILE *fp_b = fopen(biasfile, "w");
            for (k = 0; k < l->n; k = k + 1) {
                uint16_t first = k < l->n ? l->biases_quant[k] : 0;
                //uint16_t second = k+1 < l->n ? l->biases_quant[k+1] : 0;
                fprintf(fp_b, "%04x\n", first);
            }
            fclose(fp_b);

            FILE *fp_s = fopen(scalefile, "w");
            uint32_t first = l->weights_quant_multiplier;
            fprintf(fp_s, "%04x\n", first);
            fclose(fp_s);
        }
    }
}

void save_original_model(network net) {
    int j;
    for (j = 0; j < net.n; ++j) {
        layer *l = &net.layers[j];
        if (l->type == CONVOLUTIONAL) {
            size_t const weights_size = l->size*l->size*l->c*l->n;
            size_t const filter_size = l->size*l->size*l->c;

            printf(" Saving FP32 weights for CONV%d \n", j);

            char weightfile[30];

            sprintf(weightfile, "weights_FP32/CONV%d_W.txt", j);

            int k;


            FILE *fp_w = fopen(weightfile, "w");
            
            for (k = 0; k < weights_size; k = k + 1) {
                float first = k < weights_size ? l->weights[k] : 0;
                fprintf(fp_w, "%f\n", first);
            }
            
            fclose(fp_w);
        }
    }
}