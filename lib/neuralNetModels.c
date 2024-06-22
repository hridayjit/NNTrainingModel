#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "neuralNetModels.h"
#include <time.h>

void writeToFile(double accuracy, int numLayer, struct LayerParams *layerParams) {
    FILE *file = fopen("adjustedParameters.txt", "w");
    if(file != NULL) {
        fprintf(file, "The accuracy obtained is: %lf\n\n", accuracy);
        for(int i=0; i<numLayer; i++) {
            fprintf(file, "The layer no: %d\n", (i+1));
            fprintf(file, "The adjusted weights for layer %d is:\n", (i+1));
            for(int j=0; j<layerParams[i].size; j++) {
                for(int k=0; k<layerParams[i].numNeurons; k++) {
                    fprintf(file, "%lf\t", layerParams[i].weights[j][k]);
                }
                fprintf(file, "\n");
            }
            fprintf(file, "\n");
        }

        fclose(file);
    }
    
}

struct LayerParams *ImageProcessing(int numLayer, int *numNeurons, int inputSize, double *inputArr, double *actualOutput, double learningRate, double reqdAccuracy) {
    struct ModelConstants *modelConstants = (struct ModelConstants *)malloc(sizeof(struct ModelConstants));
    struct LayerParams *layerParams = (struct LayerParams *)malloc(numLayer * sizeof(struct LayerParams));
    // struct ModelConstants modelConstants;
    modelConstants->beta1 = 0.9;
    modelConstants->beta2 = 0.999;
    modelConstants->epsilon = exp(-8);
    modelConstants->neu = 0.001; //|| 0.0001;
    modelConstants->iterations = 400;
    modelConstants->reqdAccuracy = reqdAccuracy;

    for(int i=0; i<numLayer; i++) {
        layerParams[i].layerNum = (i+1);
        layerParams[i].numNeurons = numNeurons[i];
        layerParams[i].learningRate = learningRate;
        if(i == 0) {
            layerParams[i].size = inputSize;
            layerParams[i].inputArr = (double *)malloc(layerParams[i].size * sizeof(double));
            layerParams[i].inputArr = inputArr;
        }
        else{
            layerParams[i].size = layerParams[i-1].numNeurons;
        }
        if(i == (numLayer-1)) {
            layerParams[i].actual = (double *)malloc(layerParams[i].numNeurons * sizeof(double));
            layerParams[i].actual = actualOutput;
        }
        layerParams[i].weights = (double **)malloc(layerParams[i].size * sizeof(double *));
        layerParams[i].biases = (double **)malloc(layerParams[i].size * sizeof(double *));
        layerParams[i].mt = (double **)malloc(layerParams[i].size * sizeof(double *));
        layerParams[i].vt = (double **)malloc(layerParams[i].size * sizeof(double *));
        for(int j=0; j<layerParams[i].size; j++) {
            layerParams[i].weights[j] = (double *)malloc(layerParams[i].numNeurons * sizeof(double));
            layerParams[i].biases[j] = (double *)malloc(layerParams[i].numNeurons * sizeof(double));
            layerParams[i].mt[j] = (double *)malloc(layerParams[i].numNeurons * sizeof(double));
            layerParams[i].vt[j] = (double *)malloc(layerParams[i].numNeurons * sizeof(double));
        }

        for(int j=0; j<layerParams[i].size; j++) {
            for(int k=0; k<layerParams[i].numNeurons; k++) {
                layerParams[i].weights[j][k] = ((double)rand() / (double)RAND_MAX) / sqrt((double)layerParams[i].numNeurons/2);
                layerParams[i].biases[j][k] = 0;
                layerParams[i].mt[j][k] = 0;
                layerParams[i].vt[j][k] = 0;
            }
        }

    }

    
    layerParams = neuralCycleInit(numLayer, layerParams);
    // for(int i=0; i<numLayer; i++) {
    //     printf("The weights after layer %d: \n", (i+1));
    //     for(int j=0; j<layerParams[i].size; j++) {
    //         for(int k=0; k<layerParams[i].numNeurons; k++) {
    //             printf("%lf\t", firstHalfCycle[i].weights[j][k]);
    //         }
    //         printf("\n");
    //     }
    // }
    printf("The outputs are: \n");
    for(int i=0; i<layerParams[numLayer-1].numNeurons; i++) {
        printf("%lf\t", layerParams[numLayer-1].output[i]);
    }
    printf("\n");

    printf("The actual are: \n");
    for(int i=0; i<layerParams[numLayer-1].numNeurons; i++) {
        printf("%lf\t", layerParams[numLayer-1].actual[i]);
    }
    printf("\n");

    printf("---------------------------\n");

    layerParams = neuralCycle(numLayer, layerParams, modelConstants);

    

    

    // printf("The actual are: \n");
    // for(int i=0; i<layerParams[numLayer-1].numNeurons; i++) {
    //     printf("%lf\t", layerParams[numLayer-1].actual[i]);
    // }
    // printf("\n");

    return layerParams;
}

struct LayerParams *neuralCycleInit(int numLayer, struct LayerParams *layerParams) {
    struct LayerParams *layerInit = (struct LayerParams *)malloc(numLayer * sizeof(struct LayerParams));
    layerInit = forwardPropagation(numLayer, layerParams);
    return layerInit;
}

struct LayerParams *neuralCycle(int numLayer, struct LayerParams *layerParams, struct ModelConstants *modelConstants) {
    double accuracy = 0;
    int iterations = 0;
    while(accuracy < modelConstants->reqdAccuracy) {
        layerParams = backwardPropagation(numLayer, layerParams, modelConstants);
        layerParams = forwardPropagation(numLayer, layerParams);
        double sum = 0;
        for(int j=0; j<layerParams[numLayer-1].numNeurons; j++) {
            double min = 0;
            double max = 0;
            if(layerParams[numLayer-1].output[j] <= layerParams[numLayer-1].actual[j]) {
                min = layerParams[numLayer-1].output[j];
                max = layerParams[numLayer-1].actual[j];
            }
            else{
                min = layerParams[numLayer-1].actual[j];
                max = layerParams[numLayer-1].output[j];
            }
            sum += min/max; 
        }
        accuracy = sum/layerParams[numLayer-1].numNeurons;
        iterations++;
        printf("The outputs are: \n");
        for(int j=0; j<layerParams[numLayer-1].numNeurons; j++) {
            printf("%lf\t", layerParams[numLayer-1].output[j]);
        }
        printf("\n");

    }

    writeToFile(accuracy, numLayer, layerParams);

    // struct AdjustedParameters *adjustedParams = (struct AdjustedParameters *)malloc(sizeof(struct AdjustedParameters));

    // adjustedParams->accuracy = accuracy;
    // adjustedParams->layerNum = 

    printf("The accuracy is: %lf\n", accuracy);
    printf("The no. of iteration is: %d\n", iterations);
    
    // for(int i=0; i<modelConstants->iterations; i++) {
    //     layerParams = backwardPropagation(numLayer, layerParams, modelConstants);
    //     layerParams = forwardPropagation(numLayer, layerParams);
    //     double sum = 0;
    //     for(int j=0; j<layerParams[numLayer-1].numNeurons; j++) {
    //         sum += (layerParams[numLayer-1].output[j]/layerParams[numLayer-1].actual[j]);
    //     }
        
    //     double accuracy = sum/layerParams[numLayer-1].numNeurons;
    //     printf("The outputs are: \n");
    //     for(int j=0; j<layerParams[numLayer-1].numNeurons; j++) {
    //         printf("%lf\t", layerParams[numLayer-1].output[j]);
    //     }
    //     printf("\n");

    //     printf("The accuracy is: %lf\n", accuracy);
    // }
    return layerParams;
}

struct LayerParams *forwardPropagation(int numLayer, struct LayerParams *layerParams) {
    for(int i=1; i<=numLayer; i++) {
        double **weights = layerParams[i-1].weights;
        double **biases = layerParams[i-1].biases;
        int numNeurons = layerParams[i-1].numNeurons;
        double *inputArr = layerParams[i-1].inputArr;
        int size = layerParams[i-1].size;

        double* output = forwardLayer(numNeurons, inputArr, weights, biases, size);
        //previous layer leftover params
        layerParams[i-1].output = (double *)malloc(layerParams[i-1].numNeurons * sizeof(double));
        layerParams[i-1].output = output;
        if (i < numLayer) {
            layerParams[i].inputArr = (double *)malloc(layerParams[i].size * sizeof(double));
            layerParams[i].inputArr = output;
        }
    }
    return layerParams;
}

struct LayerParams *backwardPropagation(int numLayer, struct LayerParams *layerParams, struct ModelConstants *modelConstants) {
    for(int i=(numLayer-1); i>=0; i--) {
        double *cost = (double *)malloc(layerParams[i].numNeurons * sizeof(double));

        if(i == (numLayer-1)) {
            for(int j=0; j<layerParams[i].numNeurons; j++) {
                cost[j] = (layerParams[i].output[j] - layerParams[i].actual[j]);
            }
        }
        else{
            for(int j=0; j<layerParams[i].numNeurons; j++) {
                double sum = 0;
                for(int k=0; k<layerParams[i+1].numNeurons; k++) {
                    sum += (layerParams[i+1].cost[k] * layerParams[i+1].weights[j][k]);
                }
                cost[j] = layerParams[i].output[j] * (1 - layerParams[i].output[j]) * sum;
            }
        }

        for(int j=0; j<layerParams[i].size; j++) {
            for(int k=0; k<layerParams[i].numNeurons; k++) {
                double delta = cost[k] * layerParams[i].inputArr[j];
                layerParams[i].mt[j][k] = (modelConstants->beta1 * layerParams[i].mt[j][k]) + ((1-modelConstants->beta1) * delta);
                layerParams[i].vt[j][k] = (modelConstants->beta2 * layerParams[i].vt[j][k]) + ((1-modelConstants->beta2) * pow(delta, 2));
            }
        }

        double **updatedWeights = (double **)malloc(layerParams[i].size * sizeof(double *));
        for(int j=0; j<layerParams[i].size; j++) {
            updatedWeights[j] = (double *)malloc(layerParams[i].numNeurons * sizeof(double));
        }
        updatedWeights = backwardLayer(layerParams[i].numNeurons, cost, layerParams[i].learningRate, layerParams[i].inputArr, layerParams[i].weights, layerParams[i].biases, layerParams[i].size, layerParams[i].mt, layerParams[i].vt, modelConstants);
        layerParams[i].weights = updatedWeights;
        layerParams[i].cost = cost;

        //freeing memory space
        // free(cost);
        // for(int j=0; j<layerParams[i].size; j++) {
        //     free(updatedWeights[j]);
        // }
        // free(updatedWeights);
    }
    return layerParams;
}

double* forwardLayer(int numNeurons, double* inputArr, double** weights, double** biases, int size) {
    double* outputs = (double *)malloc(numNeurons * sizeof(double));
    for(int j=0; j<numNeurons; j++) {
        double* weight1 = (double *)malloc(size * sizeof(double));
        double* biases1 = (double *)malloc(size * sizeof(double));
        for(int i=0; i<size; i++) {
            weight1[i] = weights[i][j];
            biases1[i] = biases[i][j];
        }
        double output = forwardPerceptron(inputArr, weight1, biases1, size);
        outputs[j] = output;
        //freeing memory space
        // free(weight1);
        // free(biases1);
    }
    return outputs;
}

double** backwardLayer(int numNeurons, double *cost, double learningRate, double *inputArr, double** weights, double** biases, int size, double **mt, double **vt, struct ModelConstants *modelConstants) {
    //allocate memory
    double** newWeightsUnprocessed = (double **)malloc(numNeurons * sizeof(double *));
    double** newWeights = (double **)malloc(size * sizeof(double *));
    for(int i=0; i<numNeurons; i++) {
        newWeightsUnprocessed[i] = (double *)malloc(size * sizeof(double));
    }
    for(int i=0; i<size; i++) {
        newWeights[i] = (double *)malloc(numNeurons * sizeof(double));
    }
    //compute
    for(int j=0; j<numNeurons; j++) {
        double* weight1 = (double *)malloc(size * sizeof(double));
        double* biases1 = (double *)malloc(size * sizeof(double));
        double* mt1 = (double *)malloc(size * sizeof(double));
        double* vt1 = (double *)malloc(size * sizeof(double));


        for(int i=0; i<size; i++) {
            weight1[i] = weights[i][j];
            biases1[i] = biases[i][j];
            mt1[i] = mt[i][j];
            vt1[i] = vt[i][j];
        }
        newWeightsUnprocessed[j] = backwardPerceptron(weight1, biases1, size, mt1, vt1, modelConstants);//size
        // free(weight1);
        // free(biases1);
    }

    //transpose of newWeightsUnprocessed to return
    for(int i=0; i<size; i++) {
        for(int j=0; j<numNeurons; j++) {
            newWeights[i][j] = newWeightsUnprocessed[j][i];
        }
    }

    for(int i=0; i<numNeurons; i++) {
        free(newWeightsUnprocessed[i]);
    }
    free(newWeightsUnprocessed);

    return newWeights;
}

double forwardPerceptron(double* inputArr, double* weights, double* biases, int size) {
    double output;
    double weightedSum = 0;

    for(int i=0; i<size; i++) {
        weightedSum += (inputArr[i] * weights[i]) + biases[i];
    }

    output = 1/(1 + exp(-1 * weightedSum));
    return output;
}



double* backwardPerceptron(double* weights, double* biases, int size, double *mt, double *vt, struct ModelConstants *modelConstants) {
    double* weightsNew = (double *)malloc(size * sizeof(double));
    double* biasesNew = (double *)malloc(size * sizeof(double));

    for(int i=0; i<size; i++) {
        double mtcap = (mt[i] / (1-modelConstants->beta1));
        double vtcap = (vt[i] / (1-modelConstants->beta2));

        double deltaWeight = modelConstants->neu * (mtcap / sqrt(vtcap + modelConstants->epsilon));  
        weightsNew[i] = weights[i] - deltaWeight;
    } 

    return weightsNew;
}