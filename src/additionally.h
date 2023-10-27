#pragma once
#ifndef ADDITIONALLY_H
#define ADDITIONALLY_H

#include "box.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

    extern int gpu_index;

    // -------------- im2col.h --------------

    // im2col.c
    float im2col_get_pixel(float *im, int height, int width, int channels,
        int row, int col, int channel, int pad);

    // im2col.c
    //From Berkeley Vision's Caffe!
    //https://github.com/BVLC/caffe/blob/master/LICENSE
    void im2col_cpu(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col);


    // --------------  activations.h --------------

    typedef enum {
        LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN
    }ACTIVATION;

    static inline float stair_activate(float x)
    {
        int n = floor(x);
        if (n % 2 == 0) return floor(x / 2.);
        else return (x - n) + floor(x / 2.);
    }
    static inline float hardtan_activate(float x)
    {
        if (x < -1) return -1;
        if (x > 1) return 1;
        return x;
    }
    static inline float linear_activate(float x) { return x; }
    static inline float logistic_activate(float x) { return 1. / (1. + exp(-x)); }
    static inline float loggy_activate(float x) { return 2. / (1. + exp(-x)) - 1; }
    static inline float relu_activate(float x) { return x*(x>0); }
    static inline float elu_activate(float x) { return (x >= 0)*x + (x < 0)*(exp(x) - 1); }
    static inline float relie_activate(float x) { return (x>0) ? x : .01*x; }
    static inline float ramp_activate(float x) { return x*(x>0) + .1*x; }
    static inline float leaky_activate(float x) { return (x>0) ? x : .1*x; }
    static inline float tanh_activate(float x) { return (exp(2 * x) - 1) / (exp(2 * x) + 1); }
    static inline float plse_activate(float x)
    {
        if (x < -4) return .01 * (x + 4);
        if (x > 4)  return .01 * (x - 4) + 1;
        return .125*x + .5;
    }

    static inline float lhtan_activate(float x)
    {
        if (x < 0) return .001*x;
        if (x > 1) return .001*(x - 1) + 1;
        return x;
    }

    static inline ACTIVATION get_activation(char *s)
    {
        if (strcmp(s, "logistic") == 0) return LOGISTIC;
        if (strcmp(s, "loggy") == 0) return LOGGY;
        if (strcmp(s, "relu") == 0) return RELU;
        if (strcmp(s, "elu") == 0) return ELU;
        if (strcmp(s, "relie") == 0) return RELIE;
        if (strcmp(s, "plse") == 0) return PLSE;
        if (strcmp(s, "hardtan") == 0) return HARDTAN;
        if (strcmp(s, "lhtan") == 0) return LHTAN;
        if (strcmp(s, "linear") == 0) return LINEAR;
        if (strcmp(s, "ramp") == 0) return RAMP;
        if (strcmp(s, "leaky") == 0) return LEAKY;
        if (strcmp(s, "tanh") == 0) return TANH;
        if (strcmp(s, "stair") == 0) return STAIR;
        fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
        return RELU;
    }

    static float activate(float x, ACTIVATION a)
    {
        switch (a) {
        case LINEAR:
            return linear_activate(x);
        case LOGISTIC:
            return logistic_activate(x);
        case LOGGY:
            return loggy_activate(x);
        case RELU:
            return relu_activate(x);
        case ELU:
            return elu_activate(x);
        case RELIE:
            return relie_activate(x);
        case RAMP:
            return ramp_activate(x);
        case LEAKY:
            return leaky_activate(x);
        case TANH:
            return tanh_activate(x);
        case PLSE:
            return plse_activate(x);
        case STAIR:
            return stair_activate(x);
        case HARDTAN:
            return hardtan_activate(x);
        case LHTAN:
            return lhtan_activate(x);
        }
        return 0;
    }

    static void activate_array(float *x, const int n, const ACTIVATION a)
    {
        int i;
        for (i = 0; i < n; ++i) {
            x[i] = activate(x[i], a);
        }
    }

    // -------------- blas.h --------------

    // blas.c
    void gemm_nn(int M, int N, int K, float ALPHA,
        float *A, int lda, float *B, int ldb, float *C, int ldc);

    // blas.c
    void fill_cpu(int N, float ALPHA, float *X, int INCX);

    void transpose_bin(uint32_t *A, uint32_t *B, const int n, const int m,
        const int lda, const int ldb, const int block_size);

    // 32 channels -> 1 channel (with 32 floats)
    // 256 channels -> 8 channels (with 32 floats)
    void repack_input(float *input, float *re_packed_input, int w, int h, int c);

    // transpose uint32_t matrix
    void transpose_uint32(uint32_t *src, uint32_t *dst, int src_h, int src_w, int src_align, int dst_align);

    // convolution repacked bit matrix (32 channels -> 1 uint32_t) XNOR-net
    void convolution_repacked(uint32_t *packed_input, uint32_t *packed_weights, float *output,
        int w, int h, int c, int n, int size, int pad, int new_lda, float *mean_arr);

    // AVX2
    void gemm_nn_bin_32bit_packed(int M, int N, int K, float ALPHA,
        uint32_t *A, int lda,
        uint32_t *B, int ldb,
        float *C, int ldc, float *mean_arr);

    // AVX2
    void gemm_nn_bin_transposed_32bit_packed(int M, int N, int K, float ALPHA,
        uint32_t *A, int lda,
        uint32_t *B, int ldb,
        float *C, int ldc, float *mean_arr);

    // AVX2
    void im2col_cpu_custom(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col);

    // AVX2
    void im2col_cpu_custom_bin(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col, int bit_align);

    // AVX2
    void activate_array_cpu_custom(float *x, const int n, const ACTIVATION a);

    // AVX2
    void forward_maxpool_layer_avx(float *src, float *dst, int *indexes, int size, int w, int h, int out_w, int out_h, int c,
        int pad, int stride, int batch);

    // AVX2
    void float_to_bit(float *src, unsigned char *dst, size_t size);

    // AVX2
    void gemm_nn_custom_bin_mean_transposed(int M, int N, int K, float ALPHA_UNUSED,
        unsigned char *A, int lda,
        unsigned char *B, int ldb,
        float *C, int ldc, float *mean_arr);

    // -------------- list.h --------------


    typedef struct node {
        void *val;
        struct node *next;
        struct node *prev;
    } node;

    typedef struct list {
        int size;
        node *front;
        node *back;
    } list;


    // list.c
    list *get_paths(char *filename);

    // list.c
    void **list_to_array(list *l);

    // list.c
    void free_node(node *n);

    // list.c
    void free_list(list *l);

    // list.c
    char **get_labels(char *filename);


    // -------------- utils.h --------------

#define TWO_PI 6.2831853071795864769252866

    // utils.c
    void error(const char *s);

    // utils.c
    void malloc_error();

    // utils.c
    void file_error(char *s);

    // utils.c
    char *fgetl(FILE *fp);

    // utils.c
    int *read_map(char *filename);

    // utils.c
    void del_arg(int argc, char **argv, int index);

    // utils.c
    int find_arg(int argc, char* argv[], char *arg);

    // utils.c
    int find_int_arg(int argc, char **argv, char *arg, int def);

    // utils.c
    float find_float_arg(int argc, char **argv, char *arg, float def);

    // utils.c
    char *find_char_arg(int argc, char **argv, char *arg, char *def);

    // utils.c
    void strip(char *s);

    // utils.c
    void list_insert(list *l, void *val);

    // utils.c
    float rand_uniform(float min, float max);

    // utils.c
    float rand_scale(float s);

    // utils.c
    int rand_int(int min, int max);

    // utils.c
    int constrain_int(int a, int min, int max);

    // utils.c
    float dist_array(float *a, float *b, int n, int sub);

    // utils.c
    float mag_array(float *a, int n);

    // utils.c
    int max_index(float *a, int n);

    // utils.c
    // From http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    float rand_normal();

    // utils.c
    void free_ptrs(void **ptrs, int n);

    // --------------  tree.h --------------

    typedef struct {
        int *leaf;
        int n;
        int *parent;
        int *group;
        char **name;

        int groups;
        int *group_size;
        int *group_offset;
    } tree;

    // tree.c
    void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves);

    // -------------- layer.h --------------

    struct network_state;

    struct layer;
    typedef struct layer layer;

    typedef enum {
        CONVOLUTIONAL,
        DECONVOLUTIONAL,
        CONNECTED,
        MAXPOOL,
        SOFTMAX,
        DETECTION,
        DROPOUT,
        CROP,
        ROUTE,
        COST,
        NORMALIZATION,
        AVGPOOL,
        LOCAL,
        SHORTCUT,
        ACTIVE,
        RNN,
        GRU,
        CRNN,
        BATCHNORM,
        NETWORK,
        XNOR,
        REGION,
        YOLO,
        UPSAMPLE,
        REORG,
        BLANK
    } LAYER_TYPE;

    typedef enum {
        SSE, MASKED, SMOOTH
    } COST_TYPE;

    struct layer {
        LAYER_TYPE type;
        ACTIVATION activation;
        COST_TYPE cost_type;
        void(*forward)   (struct layer, struct network_state);
        int batch_normalize;
        int shortcut;
        int batch;
        int forced;
        int flipped;
        int inputs;
        int outputs;
        int truths;
        int h, w, c;
        int out_h, out_w, out_c;
        int n;
        int max_boxes;
        int groups;
        int group_id;  //NXT ++ For yolov4
        int size;
        int side;
        int stride;
        int reverse;
        int pad;
        int sqrt;
        int flip;
        int index;
        int binary;
        int steps;
        int hidden;
        float dot;
        float angle;
        float jitter;
        float saturation;
        float exposure;
        float shift;
        float ratio;
        int focal_loss;
        int softmax;
        int classes;
        int coords;
        int background;
        int rescore;
        int objectness;
        int does_cost;
        int joint;
        int noadjust;
        int reorg;
        int log;

        int *mask;
        int total;
        float bflops;

        int adam;
        float B1;
        float B2;
        float eps;
        float *m_gpu;
        float *v_gpu;
        int t;
        float *m;
        float *v;

        tree *softmax_tree;
        int  *map;

        float alpha;
        float beta;
        float kappa;

        float coord_scale;
        float object_scale;
        float noobject_scale;
        float class_scale;
        int bias_match;
        int random;
        float ignore_thresh;
        float truth_thresh;
        float thresh;
        int classfix;
        int absolute;

        int dontload;
        int dontloadscales;

        float temperature;
        float probability;
        float scale;

        int *indexes;
        float *rand;
        float *cost;
        char  *cweights;
        float *state;
        float *prev_state;
        float *forgot_state;
        float *forgot_delta;
        float *state_delta;

        float *concat;
        float *concat_delta;

        char *align_bit_weights_gpu;
        float *mean_arr_gpu;
        float *align_workspace_gpu;
        float *transposed_align_workspace_gpu;
        int align_workspace_size;

        char *align_bit_weights;
        float *mean_arr;
        int align_bit_weights_size;
        int lda_align;
        int new_lda;
        int bit_align;

        float *biases;
        float *biases_quant;

        int quantized;

        float *scales;

        float *weights;
        int8_t *weights_int8;

        float weights_quant_multiplier;
        float input_quant_multiplier;

        float *col_image;
        int   * input_layers;
        int   * input_sizes;

        float * output;
        int output_pinned;

        float output_multiplier;
        int8_t * output_int8;
        float * squared;
        float * norms;

        float * spatial_mean;
        float * mean;
        float * variance;

        float * rolling_mean;
        float * rolling_variance;

        float * x;
        float * x_norm;

        struct layer *input_layer;
        struct layer *self_layer;
        struct layer *output_layer;

        struct layer *input_gate_layer;
        struct layer *state_gate_layer;
        struct layer *input_save_layer;
        struct layer *state_save_layer;
        struct layer *input_state_layer;
        struct layer *state_state_layer;

        struct layer *input_z_layer;
        struct layer *state_z_layer;

        struct layer *input_r_layer;
        struct layer *state_r_layer;

        struct layer *input_h_layer;
        struct layer *state_h_layer;

        float *z_cpu;
        float *r_cpu;
        float *h_cpu;

        float *binary_input;
        uint32_t *bin_re_packed_input;
        char *t_bit_input;

        size_t workspace_size;
    };

    typedef layer local_layer;
    typedef layer convolutional_layer;
    typedef layer softmax_layer;
    typedef layer region_layer;
    typedef layer reorg_layer;
    typedef layer maxpool_layer;
    typedef layer route_layer;

    void free_layer(layer);


    // -------------- network.h --------------

    typedef enum {
        CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
    } learning_rate_policy;

    typedef struct network {
        int quantized;
        float *workspace;
        int n;
        int batch;
        float *input_calibration;
        int input_calibration_size;
        uint64_t *seen;
        float epoch;
        int subdivisions;
        float momentum;
        float decay;
        layer *layers;
        int outputs;
        float *output;
        learning_rate_policy policy;

        float learning_rate;
        float gamma;
        float scale;
        float power;
        int time_steps;
        int step;
        int max_batches;
        float *scales;
        int   *steps;
        int num_steps;
        int burn_in;

        int adam;
        float B1;
        float B2;
        float eps;

        int inputs;
        int h, w, c;
        int max_crop;
        int min_crop;
        float angle;
        float aspect;
        float exposure;
        float saturation;
        float hue;

        int gpu_index;
        tree *hierarchy;
        int do_input_calibration;
    } network;

    typedef struct network_state {
        float *truth;
        float *input;
        int8_t *input_int8;
        float *delta;
        float *workspace;
        int train;
        int index;
        network net;
    } network_state;


    // network.c
    network make_network(int n);

    // network.c
    void set_batch_network(network *net, int b);


    // -------------- softmax_layer.h --------------

    // softmax_layer.c
    softmax_layer make_softmax_layer(int batch, int inputs, int groups);

    // -------------- reorg_layer.h --------------

    // reorg_layer.c
    layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse);

    // -------------- route_layer.h --------------

    // route_layer.c
    // NXT ++: Add for yolov4 or later
    //route_layer make_route_layer(int batch, int n, int *input_layers, int *input_sizes);
    route_layer make_route_layer(int batch, int n, int *input_layers, int *input_sizes, int groups, int group_id);

    // -------------- region_layer.h --------------

    //  region_layer.c
    region_layer make_region_layer(int batch, int w, int h, int n, int classes, int coords);


    // -------------- maxpool_layer.h --------------

    // maxpool_layer.c
    maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding);

    // -------------- convolutional_layer.h --------------

    // convolutional_layer.c
    convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int adam, int quantized);



    // -------------- image.c --------------

    // image.c
    typedef struct {
        int h;
        int w;
        int c;
        float *data;
    } image;

    // image.c
    void rgbgr_image(image im);

    // image.c
    image make_empty_image(int w, int h, int c);

    // image.c
    void free_image(image m);

    // image.c
    void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b);

    // image.c
    void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);

    // image.c
    image make_image(int w, int h, int c);

    // image.c
    float get_pixel(image m, int x, int y, int c);

    // image.c
    void set_pixel(image m, int x, int y, int c, float val);

    // image.c
    image resize_image(image im, int w, int h);

    // Added
    image to_square_image(image im);

    // image.c
    image load_image(char *filename, int w, int h, int c, int zero_center);

    // image.c
    image load_image_stb(char *filename, int channels, int zero_center);

    // image.c
    image load_image_cv(char *filename, int channels);

    // image.c
    float get_color(int c, int x, int max);

    // image.c
    void save_image_png(image im, const char *name, int zero_center);

    // image.c
    void show_image(image p, const char *name, int zero_center);


    // -------------- parser.c --------------------

    // parser.c
    network parse_network_cfg(char *filename, int batch, int quantized);

    // parser.c
    void load_weights_upto_cpu(network *net, char *filename, int cutoff);


    // -------------- yolov2_forward_network.c --------------------
    void forward_maxpool_layer_cpu(const layer l, network_state state);

    void forward_route_layer_cpu(const layer l, network_state state);

    void forward_reorg_layer_cpu(const layer l, network_state state);

    void forward_upsample_layer_cpu(const layer l, network_state net);

    void forward_shortcut_layer_cpu(const layer l, network_state state);

    void forward_yolo_layer_cpu(const layer l, network_state state);

    void forward_region_layer_cpu(const layer l, network_state state);

    // detect on CPU: yolov2_forward_network.c
    float *network_predict_cpu(network net, float *input);

    // calculate mAP
    void validate_detector_map(char *datacfg, char *cfgfile, char *weightfile, float thresh_calc_avg_iou, int quantized, int save_params, int zero_center, const float iou_thresh);

    // fuse convolutional and batch_norm weights into one convolutional-layer
    void yolov2_fuse_conv_batchnorm(network net);

    // -------------- yolov2_forward_network_quantized.c --------------------

    // yolov2_forward_network.c - fp32 is used for the last layer during INT8-quantized inference
    void forward_convolutional_layer_cpu(layer l, network_state state);

    // quantized
    float *network_predict_quantized(network net, float *input, int save_interim_result);

    // get multipliers for layer input, convolutional weights and biases and quantize the weights and biases
    void do_quantization(network net);

    // save quantized model
    void save_quantized_model(network net);

    // save original model
    void save_original_model(network net);

    // draw distribution of float values
    void draw_distribution(float *arr_ptr, int arr_size, char *name);

    // additionally.c
    detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter);

    // additionally.c
    int entry_index(layer l, int batch, int location, int entry);

    // additionally.c
    void free_detections(detection *dets, int n);

    // -------------- gettimeofday for Windows--------------------

#if defined(_MSC_VER)
#include <time.h>
#include <windows.h> //I've ommited this line.
#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#include <sys/time.h>
#endif

    struct timezone
    {
        int  tz_minuteswest; /* minutes W of Greenwich */
        int  tz_dsttime;     /* type of dst correction */
    };

    int gettimeofday(struct timeval *tv, struct timezone *tz);
#endif

#ifdef __cplusplus
}
#endif

#endif    // ADDITIONALLY_H