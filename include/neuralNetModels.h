#ifndef NEURALNETMODELS_H
#define NEURALNETMODELS_H

// #include <stdio.h>
// #include <stdlib.h>
// #include <math.h>
// #include "neuralNetModels.h"
// #include <time.h>

struct LayerParams
{
    int layerNum;
    int numNeurons;
    int size;
    double learningRate;
    double *inputArr;
    double **weights;
    double **biases;
    double *output;
    double *actual;
    double *cost;
    double **mt;
    double **vt;
};

struct ModelConstants
{
    double beta1;
    double beta2;
    double epsilon;
    double neu;
    int iterations;
    double reqdAccuracy;
};

struct AdjustedParameters
{
    int layerNum;
    double accuracy;
    double **weights;
};


struct LayerParams *ImageProcessing(int numLayer, int *numNeurons, int inputSize, double *inputArr, double *actualOutput, double learningRate, double actualAccuracy);
struct LayerParams *neuralCycleInit(int numLayer, struct LayerParams *layerParams);
struct LayerParams *neuralCycle(int numLayer, struct LayerParams *layerParams, struct ModelConstants *modelConstants);
struct LayerParams *forwardPropagation(int numLayer, struct LayerParams *layerParams);
double* forwardLayer(int numNeurons, double* inputArr, double** weights, double** biases, int size);
double forwardPerceptron(double* inputArr, double* weights, double* biases, int size);
struct LayerParams *backwardPropagation(int numLayer, struct LayerParams *layerParams, struct ModelConstants *modelConstants);
double** backwardLayer(int numNeurons, double *cost, double learningRate, double *inputArr, double** weights, double** biases, int size, double **mt, double **vt, struct ModelConstants *modelConstants);
double* backwardPerceptron(double* weights, double* biases, int size, double *mt, double *vt, struct ModelConstants *modelConstants);


#endif