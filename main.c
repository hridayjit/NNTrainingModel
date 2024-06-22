#include <stdio.h>
#include <stdlib.h>
#include "neuralNetModels.h"

int main() {
    const double learningRate = 0.01;
    const double reqdAccuracy = 0.99;
    int numLayer;
    printf("Enter the number of layers (excluding input layer) in the model:\n");
    scanf("%d", &numLayer);

    int *numNeurons = (int *)malloc(numLayer * sizeof(int));
    for(int i=0; i<numLayer; i++) {
        printf("Enter the no. of neurons in Layer %d\n", (i+1));
        scanf("%d", &numNeurons[i]);
        printf("\n");
    }

    int size;
    printf("Enter the size of the input vector: \n");
    scanf("%d", &size);
    printf("\n");

    double *inputArr = (double *)malloc(size * sizeof(double));
    printf("Enter the values of the input vector:\n");
    for(int i=0; i<size; i++) {
        scanf("%lf", &inputArr[i]);
        printf("\n");
    }

    double *actual = (double *)malloc(numNeurons[numLayer-1] * sizeof(double));
    printf("Enter the values of the actual vector:\n");
    for(int i=0; i<numNeurons[numLayer-1]; i++) {
        scanf("%lf", &actual[i]);
        printf("\n");
    }

    struct LayerParams *layerParams = (struct LayerParams *)malloc(numLayer * sizeof(struct LayerParams));

    layerParams = ImageProcessing(numLayer, numNeurons, size, inputArr, actual, learningRate, reqdAccuracy);
    

    return 0;
}