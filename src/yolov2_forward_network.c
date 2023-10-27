#include "additionally.h"    // some definitions from: im2col.h, blas.h, list.h, utils.h, activations.h, tree.h, layer.h, network.h
// softmax_layer.h, reorg_layer.h, route_layer.h, region_layer.h, maxpool_layer.h, convolutional_layer.h

#define GEMMCONV

// 4 layers in 1: convolution, batch-normalization, BIAS and activation
void forward_convolutional_layer_cpu(layer l, network_state state)
{

    int out_h = (l.h + 2 * l.pad - l.size) / l.stride + 1;    // output_height=input_height for stride=1 and pad=1
    int out_w = (l.w + 2 * l.pad - l.size) / l.stride + 1;    // output_width=input_width for stride=1 and pad=1
    int i, f, j;

    // fill zero (ALPHA)
    for (i = 0; i < l.outputs*l.batch; ++i) l.output[i] = 0;

    // l.n - number of filters on this layer
    // l.c - channels of input-array
    // l.h - height of input-array
    // l.w - width of input-array
    // l.size - width and height of filters (the same size for all filters)

    char input[50];
    snprintf(input, sizeof(input), "interim_result_FP32/%d_CONV_0_input.txt", state.index);
    FILE *fp_i = fopen(input, "w");
    for (int x = 0; x < l.h*l.w; x = x + 1) {
        for (int y = 0; y < l.c; y = y + 1) {
            int idx = x + y*l.h*l.w;
            float first = state.input[idx];
            fprintf(fp_i, "%f\n", first);
        }
    }
    fclose(fp_i);
    
    // 1. Convolution
    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = out_h*out_w;
    float *a = l.weights;
    float *b = state.workspace;
    float *c = l.output;



    // convolution as GEMM (as part of BLAS)
    for (i = 0; i < l.batch; ++i) {
        im2col_cpu_custom(state.input, l.c, l.h, l.w, l.size, l.stride, l.pad, b);    // AVX2
        int t;
        #pragma omp parallel for
        for (t = 0; t < m; ++t) {
            gemm_nn(1, n, k, 1, a + t*k, k, b, n, c + t*n, n);
        }
        c += n*m;
        state.input += l.c*l.h*l.w;

    }

    int const out_size = out_h*out_w;

    // 2. Batch normalization
    if (l.batch_normalize) {
        int b;
        for (b = 0; b < l.batch; b++) {
            for (f = 0; f < l.out_c; ++f) {
                for (i = 0; i < out_size; ++i) {
                    int index = f*out_size + i;
                    l.output[index+b*l.outputs] = (l.output[index+b*l.outputs] - l.rolling_mean[f]) / (sqrtf(l.rolling_variance[f]) + .000001f);
                }
            }

            // scale_bias
            for (i = 0; i < l.out_c; ++i) {
                for (j = 0; j < out_size; ++j) {
                    l.output[i*out_size + j+b*l.outputs] *= l.scales[i];
                }
            }
        }
    }

    // 3. Add BIAS
    {
        int b;
        for (b = 0; b < l.batch; b++) {
            for (i = 0; i < l.n; ++i) {
                for (j = 0; j < out_size; ++j) {
                    l.output[i*out_size + j + b*l.outputs] += l.biases[i];
                }
            }
        }
    }   

    // 4. Activation function (LEAKY or LINEAR)
    activate_array_cpu_custom(l.output, l.outputs*l.batch, l.activation);

}



// MAX pooling layer
void forward_maxpool_layer_cpu(const layer l, network_state state)
{
    if (!state.train) {
        forward_maxpool_layer_avx(state.input, l.output, l.indexes, l.size, l.w, l.h, l.out_w, l.out_h, l.c, l.pad, l.stride, l.batch);
        return;
    }

    int b, i, j, k, m, n;
    const int w_offset = -l.pad;
    const int h_offset = -l.pad;

    const int h = l.out_h;
    const int w = l.out_w;
    const int c = l.c;

    // batch index
    for (b = 0; b < l.batch; ++b) {
        // channel index
        for (k = 0; k < c; ++k) {
            // y - input
            for (i = 0; i < h; ++i) {
                // x - input
                for (j = 0; j < w; ++j) {
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    // pooling x-index
                    for (n = 0; n < l.size; ++n) {
                        // pooling y-index
                        for (m = 0; m < l.size; ++m) {
                            int cur_h = h_offset + i*l.stride + n;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                cur_w >= 0 && cur_w < l.w);
                            float val = (valid != 0) ? state.input[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;    // get max index
                            max = (val > max) ? val : max;            // get max value
                        }
                    }
                    l.output[out_index] = max;        // store max value
                    l.indexes[out_index] = max_i;    // store max index
                }
            }
        }
    }
}


// // Route layer - concatenate outputs of select previous layers
// void forward_route_layer_cpu(const layer l, network_state state)
// {
//     int i, j;
//     int offset = 0;
//     // number of merged layers
//     for (i = 0; i < l.n; ++i) {
//         int index = l.input_layers[i];                    // source layer index
//         float *input = state.net.layers[index].output;    // source layer output ptr
//         int input_size = l.input_sizes[i];                // source layer size
//                                                         // batch index
//         for (j = 0; j < l.batch; ++j) {
//             memcpy(l.output + offset + j*l.outputs, input + j*input_size, input_size * sizeof(float));
//         }
//         offset += input_size;
//     }
// }

// Route layer - concatenate outputs of select previous layers
void forward_route_layer_cpu(const layer l, network_state state)
{
    int i, j;
    int offset = 0;
    // number of merged layers
    for (i = 0; i < l.n; ++i) {
        int index = l.input_layers[i];                    // source layer index
        float *input = state.net.layers[index].output;    // source layer output ptr
        int input_size = l.input_sizes[i];                // source layer size
                                                        // batch index
        //NXT ++: Modify for YOLOv4     
        //{{{
        int part_input_size = input_size / l.groups;                                                
        for (j = 0; j < l.batch; ++j) {
            //memcpy(l.output + offset + j*l.outputs, input + j*input_size, input_size * sizeof(float));
            memcpy(l.output + offset + j*l.outputs, input + j*input_size + part_input_size*l.group_id, part_input_size * sizeof(float));
        }
        //offset += input_size;
        offset += part_input_size;
        //}}}
    }
}


// Reorg layer - just change dimension sizes of the previous layer (some dimension sizes are increased by decreasing other)
void forward_reorg_layer_cpu(const layer l, network_state state)
{
    float *out = l.output;
    float *x = state.input;

    int out_w = l.out_w;
    int out_h = l.out_h;
    int out_c = l.out_c;
    int batch = l.batch;
    int stride = l.stride;

    int b, i, j, k;
    int in_c = out_c / (stride*stride);

    //printf("\n out_c = %d, out_w = %d, out_h = %d, stride = %d, forward = %d \n", out_c, out_w, out_h, stride, forward);
    //printf("  in_c = %d,  in_w = %d,  in_h = %d \n", in_c, out_w*stride, out_h*stride);

    // batch
    for (b = 0; b < batch; ++b) {
        // channel
        for (k = 0; k < out_c; ++k) {
            // y
            for (j = 0; j < out_h; ++j) {
                // x
                for (i = 0; i < out_w; ++i) {
                    int in_index = i + out_w*(j + out_h*(k + out_c*b));
                    int c2 = k % in_c;
                    int offset = k / in_c;
                    int w2 = i*stride + offset % stride;
                    int h2 = j*stride + offset / stride;
                    int out_index = w2 + out_w*stride*(h2 + out_h*stride*(c2 + in_c*b));
                    out[in_index] = x[out_index];
                }
            }
        }
    }
}



// ---- upsample layer ----

// upsample_layer.c
void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    int i, j, k, b;
    for (b = 0; b < batch; ++b) {
        for (k = 0; k < c; ++k) {
            for (j = 0; j < h*stride; ++j) {
                for (i = 0; i < w*stride; ++i) {
                    int in_index = b*w*h*c + k*w*h + (j / stride)*w + i / stride;
                    int out_index = b*w*h*c*stride*stride + k*w*h*stride*stride + j*w*stride + i;
                    if (forward) out[out_index] = scale*in[in_index];
                    else in[in_index] += scale*out[out_index];
                }
            }
        }
    }
}

// upsample_layer.c
void forward_upsample_layer_cpu(const layer l, network_state net)
{
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    if (l.reverse) {
        upsample_cpu(l.output, l.out_w, l.out_h, l.c, l.batch, l.stride, 0, l.scale, net.input);
    }
    else {
        upsample_cpu(net.input, l.w, l.h, l.c, l.batch, l.stride, 1, l.scale, l.output);
    }
}

// blas.c (shortcut_layer)
void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out)
{
    int stride = w1 / w2;
    int sample = w2 / w1;
    assert(stride == h1 / h2);
    assert(sample == h2 / h1);
    if (stride < 1) stride = 1;
    if (sample < 1) sample = 1;
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int i, j, k, b;
    for (b = 0; b < batch; ++b) {
        for (k = 0; k < minc; ++k) {
            for (j = 0; j < minh; ++j) {
                for (i = 0; i < minw; ++i) {
                    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
                    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
                    out[out_index] += add[add_index];
                }
            }
        }
    }
}

// blas.c
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for (i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}

// shortcut_layer.c
void forward_shortcut_layer_cpu(const layer l, network_state state)
{
    copy_cpu(l.outputs*l.batch, state.input, 1, l.output, 1);
    shortcut_cpu(l.batch, l.w, l.h, l.c, state.net.layers[l.index].output, l.out_w, l.out_h, l.out_c, l.output);
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

// ---- yolo layer ----

void forward_yolo_layer_cpu(const layer l, network_state state)
{
    int b, n;
    memcpy(l.output, state.input, l.outputs*l.batch * sizeof(float));

    for (b = 0; b < l.batch; ++b) {
        for (n = 0; n < l.n; ++n) {
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2 * l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array(l.output + index, (1 + l.classes)*l.w*l.h, LOGISTIC);
        }
    }
}

// ---- region layer ----

static void softmax_cpu(float *input, int n, float temp, float *output)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for (i = 0; i < n; ++i) {
        if (input[i] > largest) largest = input[i];
    }
    for (i = 0; i < n; ++i) {
        float e = expf(input[i] / temp - largest / temp);
        sum += e;
        output[i] = e;
    }
    for (i = 0; i < n; ++i) {
        output[i] /= sum;
    }
}

static void softmax_tree(float *input, int batch, int inputs, float temp, tree *hierarchy, float *output)
{
    int b;
    for (b = 0; b < batch; ++b) {
        int i;
        int count = 0;
        for (i = 0; i < hierarchy->groups; ++i) {
            int group_size = hierarchy->group_size[i];
            softmax_cpu(input + b*inputs + count, group_size, temp, output + b*inputs + count);
            count += group_size;
        }
    }
}
// ---


// Region layer - just change places of array items, then do logistic_activate and softmax
void forward_region_layer_cpu(const layer l, network_state state)
{
    int i, b;
    int size = l.coords + l.classes + 1;    // 4 Coords(x,y,w,h) + Classes + 1 Probability-t0
    memcpy(l.output, state.input, l.outputs*l.batch * sizeof(float));

    //flatten(l.output, l.w*l.h, size*l.n, l.batch, 1);
    // convert many channels to the one channel (depth=1)
    // (each grid cell will have a number of float-variables equal = to the initial number of channels)
    {
        float *x = l.output;
        int layer_size = l.w*l.h;    // W x H - size of layer
        int layers = size*l.n;        // number of channels (where l.n = number of anchors)
        int batch = l.batch;

        float *swap = calloc(layer_size*layers*batch, sizeof(float));
        int i, c, b;
        // batch index
        for (b = 0; b < batch; ++b) {
            // channel index
            for (c = 0; c < layers; ++c) {
                // layer grid index
                for (i = 0; i < layer_size; ++i) {
                    int i1 = b*layers*layer_size + c*layer_size + i;
                    int i2 = b*layers*layer_size + i*layers + c;
                    swap[i2] = x[i1];
                }
            }
        }
        memcpy(x, swap, layer_size*layers*batch * sizeof(float));
        free(swap);
    }


    // logistic activation only for: t0 (where is t0 = Probability * IoU(box, object))
    for (b = 0; b < l.batch; ++b) {
        // for each item (x, y, anchor-index)
        for (i = 0; i < l.h*l.w*l.n; ++i) {
            int index = size*i + b*l.outputs;
            float x = l.output[index + 4];
            l.output[index + 4] = 1.0F / (1.0F + expf(-x));    // logistic_activate_cpu(l.output[index + 4]);
        }
    }


    if (l.softmax_tree) {    // Yolo 9000
        for (b = 0; b < l.batch; ++b) {
            for (i = 0; i < l.h*l.w*l.n; ++i) {
                int index = size*i + b*l.outputs;
                softmax_tree(l.output + index + 5, 1, 0, 1, l.softmax_tree, l.output + index + 5);
            }
        }
    }
    else if (l.softmax) {    // Yolo v2
        // softmax activation only for Classes probability
        for (b = 0; b < l.batch; ++b) {
            // for each item (x, y, anchor-index)
            for (i = 0; i < l.h*l.w*l.n; ++i) {
                int index = size*i + b*l.outputs;
                softmax_cpu(l.output + index + 5, l.classes, 1, l.output + index + 5);
            }
        }
    }

}


void yolov2_forward_network_cpu(network net, network_state state)
{
    state.workspace = net.workspace;
    int i;
    for (i = 0; i < net.n; ++i) {
        state.index = i;
        layer l = net.layers[i];

        if (l.type == CONVOLUTIONAL) {
            forward_convolutional_layer_cpu(l, state);
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
float *network_predict_cpu(network net, float *input)
{
    network_state state;
    state.net = net;
    state.index = 0;
    state.input = input;
    state.truth = 0;
    state.train = 0;
    state.delta = 0;
    yolov2_forward_network_cpu(net, state);    // network on CPU
                                            //float *out = get_network_output(net);
    int i;
    for (i = net.n - 1; i > 0; --i) if (net.layers[i].type != COST) break;
    return net.layers[i].output;
}


// --------------------
// x - last conv-layer output
// biases - anchors from cfg-file
// n - number of anchors from cfg-file
box get_region_box_cpu(float *x, float *biases, int n, int index, int i, int j, int w, int h)
{
    box b;
    b.x = (i + logistic_activate(x[index + 0])) / w;    // (col + 1./(1. + exp(-x))) / width_last_layer
    b.y = (j + logistic_activate(x[index + 1])) / h;    // (row + 1./(1. + exp(-x))) / height_last_layer
    b.w = expf(x[index + 2]) * biases[2 * n] / w;        // exp(x) * anchor_w / width_last_layer
    b.h = expf(x[index + 3]) * biases[2 * n + 1] / h;    // exp(x) * anchor_h / height_last_layer
    return b;
}

// get prediction boxes
void get_region_boxes_cpu(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map)
{
    int i;
    float *const predictions = l.output;
    // grid index
    #pragma omp parallel for
    for (i = 0; i < l.w*l.h; ++i) {
        int j, n;
        int row = i / l.w;
        int col = i % l.w;
        // anchor index
        for (n = 0; n < l.n; ++n) {
            int index = i*l.n + n;    // index for each grid-cell & anchor
            int p_index = index * (l.classes + 5) + 4;
            float scale = predictions[p_index];                // scale = t0 = Probability * IoU(box, object)
            if (l.classfix == -1 && scale < .5) scale = 0;    // if(t0 < 0.5) t0 = 0;
            int box_index = index * (l.classes + 5);
            boxes[index] = get_region_box_cpu(predictions, l.biases, n, box_index, col, row, l.w, l.h);
            boxes[index].x *= w;
            boxes[index].y *= h;
            boxes[index].w *= w;
            boxes[index].h *= h;

            int class_index = index * (l.classes + 5) + 5;

            // Yolo 9000 or Yolo v2
            if (l.softmax_tree) {
                // Yolo 9000
                hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0);
                int found = 0;
                if (map) {
                    for (j = 0; j < 200; ++j) {
                        float prob = scale*predictions[class_index + map[j]];
                        probs[index][j] = (prob > thresh) ? prob : 0;
                    }
                }
                else {
                    for (j = l.classes - 1; j >= 0; --j) {
                        if (!found && predictions[class_index + j] > .5) {
                            found = 1;
                        }
                        else {
                            predictions[class_index + j] = 0;
                        }
                        float prob = predictions[class_index + j];
                        probs[index][j] = (scale > thresh) ? prob : 0;
                    }
                }
            }
            else
            {
                // Yolo v2
                for (j = 0; j < l.classes; ++j) {
                    float prob = scale*predictions[class_index + j];    // prob = IoU(box, object) = t0 * class-probability
                    probs[index][j] = (prob > thresh) ? prob : 0;        // if (IoU < threshold) IoU = 0;
                }
            }
            if (only_objectness) {
                probs[index][0] = scale;
            }
        }
    }
}