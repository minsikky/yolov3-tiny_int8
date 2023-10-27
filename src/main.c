#include <stdio.h>
#include <stdlib.h>

#include "box.h"
#include "pthread.h"

#include "additionally.h"

// get prediction boxes: yolov2_forward_network.c
void get_region_boxes_cpu(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map);

typedef struct detection_with_class {
    detection det;
    // The most probable class id: the best class index in this->prob.
    // Is filled temporary when processing results, otherwise not initialized
    int best_class;
} detection_with_class;

// Creates array of detections with prob > thresh and fills best_class for them
detection_with_class* get_actual_detections(detection *dets, int dets_num, float thresh, int* selected_detections_num)
{
    int selected_num = 0;
    detection_with_class* result_arr = calloc(dets_num, sizeof(detection_with_class));
    int i;
    for (i = 0; i < dets_num; ++i) {
        int best_class = -1;
        float best_class_prob = thresh;
        int j;
        for (j = 0; j < dets[i].classes; ++j) {
            if (dets[i].prob[j] > best_class_prob) {
                best_class = j;
                best_class_prob = dets[i].prob[j];
            }
        }
        if (best_class >= 0) {
            result_arr[selected_num].det = dets[i];
            result_arr[selected_num].best_class = best_class;
            ++selected_num;
        }
    }
    if (selected_detections_num)
        *selected_detections_num = selected_num;
    return result_arr;
}

// compare to sort detection** by bbox.x
int compare_by_lefts(const void *a_ptr, const void *b_ptr) {
    const detection_with_class* a = (detection_with_class*)a_ptr;
    const detection_with_class* b = (detection_with_class*)b_ptr;
    const float delta = (a->det.bbox.x - a->det.bbox.w / 2) - (b->det.bbox.x - b->det.bbox.w / 2);
    return delta < 0 ? -1 : delta > 0 ? 1 : 0;
}

// compare to sort detection** by best_class probability
int compare_by_probs(const void *a_ptr, const void *b_ptr) {
    const detection_with_class* a = (detection_with_class*)a_ptr;
    const detection_with_class* b = (detection_with_class*)b_ptr;
    float delta = a->det.prob[a->best_class] - b->det.prob[b->best_class];
    return delta < 0 ? -1 : delta > 0 ? 1 : 0;
}

void draw_detections_v3(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes, int ext_output)
{
    int selected_detections_num;
    detection_with_class* selected_detections = get_actual_detections(dets, num, thresh, &selected_detections_num);

    // text output
    qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_lefts);
    int i;
    for (i = 0; i < selected_detections_num; ++i) {
        const int best_class = selected_detections[i].best_class;
        printf("%s: %.0f%%", names[best_class], selected_detections[i].det.prob[best_class] * 100);
        if (ext_output)
            printf("\t(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)\n",
                round((selected_detections[i].det.bbox.x - selected_detections[i].det.bbox.w / 2)*im.w),
                round((selected_detections[i].det.bbox.y - selected_detections[i].det.bbox.h / 2)*im.h),
                round(selected_detections[i].det.bbox.w*im.w), round(selected_detections[i].det.bbox.h*im.h));
        else
            printf("\n");
        int j;
        for (j = 0; j < classes; ++j) {
            if (selected_detections[i].det.prob[j] > thresh && j != best_class) {
                printf("%s: %.0f%%\n", names[j], selected_detections[i].det.prob[j] * 100);
            }
        }
    }

    // image output
    qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_probs);
    for (i = 0; i < selected_detections_num; ++i) {
        int width = im.h * .006;
        if (width < 1)
            width = 1;

        /*
        if(0){
        width = pow(prob, 1./2.)*10+1;
        alphabet = 0;
        }
        */

        //printf("%d %s: %.0f%%\n", i, names[selected_detections[i].best_class], prob*100);
        int offset = selected_detections[i].best_class * 123457 % classes;
        float red = get_color(2, offset, classes);
        float green = get_color(1, offset, classes);
        float blue = get_color(0, offset, classes);
        float rgb[3];

        //width = prob*20+2;

        rgb[0] = red;
        rgb[1] = green;
        rgb[2] = blue;
        box b = selected_detections[i].det.bbox;
        //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

        int left = (b.x - b.w / 2.)*im.w;
        int right = (b.x + b.w / 2.)*im.w;
        int top = (b.y - b.h / 2.)*im.h;
        int bot = (b.y + b.h / 2.)*im.h;

        if (left < 0) left = 0;
        if (right > im.w - 1) right = im.w - 1;
        if (top < 0) top = 0;
        if (bot > im.h - 1) bot = im.h - 1;

        draw_box_width(im, left, top, right, bot, width, red, green, blue);
    }
    free(selected_detections);
}



// --------------- Detect on the Image ---------------


// Detect on Image: this function uses other functions not from this file
void test_detector_cpu(char **names, char *cfgfile, char *weightfile, char *filename, float thresh, int quantized, int save_params, int zero_center, int dont_show)
{
    //image **alphabet = load_alphabet();            // image.c
    image **alphabet = NULL;
    network net = parse_network_cfg(cfgfile, 1, quantized);    // parser.c
    if (weightfile) {
        load_weights_upto_cpu(&net, weightfile, net.n);    // parser.c
    }
    //set_batch_network(&net, 1);                    // network.c
    srand(2222222);
    yolov2_fuse_conv_batchnorm(net);
    if (quantized) {
        printf("\n\n Quantization! \n\n");
        do_quantization(net);
        if (save_params) {
            printf("\n Saving quantized model... \n\n");
            save_quantized_model(net);
        }
    }

    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms = .4;
    while (1) {
        if (filename) {
            strncpy(input, filename, 256);
        }
        else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if (!input) return;
            strtok(input, "\n");
        }
        image im = load_image(input, 0, 0, 3, zero_center);            // image.c
        image sized = resize_image(im, net.w, net.h);    // image.c
        layer l = net.layers[net.n - 1];

        box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
        float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
        for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
        float *X = sized.data;
        time = clock();
        if (quantized) {
            network_predict_quantized(net, X, 1);    // quantized
            nms = 0.2;
        }
        else {
            network_predict_cpu(net, X);
        }
        printf("%s: Predicted in %f seconds.\n", input, (float)(clock() - time) / CLOCKS_PER_SEC); //sec(clock() - time));

        float hier_thresh = 0.5;
        int ext_output = 1, letterbox = 0, nboxes = 0;
        detection *dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letterbox);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        draw_detections_v3(im, dets, nboxes, thresh, names, alphabet, l.classes, ext_output);

        save_image_png(im, "predictions", zero_center);    // image.c
        if (!dont_show) {
            show_image(im, "predictions", zero_center);    // image.c
        }

        free_image(im);                    // image.c
        free_image(sized);                // image.c
        free(boxes);
        free_ptrs((void **)probs, l.w*l.h*l.n);    // utils.c
        if (filename) break;
    }
}

// get command line parameters and load objects names
void run_detector(int argc, char **argv)
{
    int dont_show = find_arg(argc, argv, "-dont_show");
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .25);
    float iou_thresh = find_float_arg(argc, argv, "-iou_thresh", .5);    // 0.5 for mAP
    char *out_filename = find_char_arg(argc, argv, "-out_filename", 0);
    int quantized = find_arg(argc, argv, "-quantized");
    int input_calibration = find_int_arg(argc, argv, "-input_calibration", 0);
    int save_params = find_arg(argc, argv, "-save_params");
    int zero_center = find_arg(argc, argv, "-zero_center");
    if (argc < 4) {
        fprintf(stderr, "usage: %s %s [map/test] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    int clear = 0;                // find_arg(argc, argv, "-clear");

    char *obj_names = argv[3];    // char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6] : 0;

    // load object names
    char **names = calloc(10000, sizeof(char *));
    int obj_count = 0;
    FILE* fp;
    char buffer[255];
    fp = fopen(obj_names, "r");
    while (fgets(buffer, 255, (FILE*)fp)) {
        names[obj_count] = calloc(strlen(buffer)+1, sizeof(char));
        strcpy(names[obj_count], buffer);
        names[obj_count][strlen(buffer) - 1] = '\0'; //remove newline
        ++obj_count;
    }
    fclose(fp);
    int classes = obj_count;

    if (0 == strcmp(argv[2], "test")) test_detector_cpu(names, cfg, weights, filename, thresh, quantized, save_params, zero_center, dont_show);
    else if (0 == strcmp(argv[2], "map")) validate_detector_map(obj_names, cfg, weights, thresh, quantized, save_params, zero_center, iou_thresh);

    int i;
    for (i = 0; i < obj_count; ++i) free(names[i]);
    free(names);
}


int main(int argc, char **argv)
{
#ifdef _DEBUG
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
    int i;
    for (i = 0; i < argc; ++i) {
        if (!argv[i]) continue;
        strip(argv[i]);
    }

    if (argc < 2) {
        fprintf(stderr, "usage: %s <function>\n", argv[0]);
        return 0;
    }
    gpu_index = find_int_arg(argc, argv, "-i", 0);  //  gpu_index = 0;

#ifndef GPU
    gpu_index = -1;
#endif
    run_detector(argc, argv);
    return 0;
}
