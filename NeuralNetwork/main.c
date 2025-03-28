//Date: 3/28/24
//This program is a simple neural network that can learn xor
//XOR is often used to test basic neural networks since it cannot be solved by
//a single layer perception
//XOR is a logical operation that compares two bits
//if the bits are different the result is 1
//if the bits are the same then the result is 0

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double sigmoid(double x){return 1 / (1+ exp(-x)); }

//takes derivative of sigmoid function
double dSigmoid(double x){return x * (1 - x); }

//when initWeights is called, init random numbers between 0 and 1
double initWeights() {return ((double)rand()) / ((double)RAND_MAX); }

//shuffles array for training, n being size of data set
void shuffle(int *array, size_t n){

    if (n > 1){
        size_t i;
        for(size_t i = 0; i < n - 1; i++){
            //create random indexes then specify new indexes to data set
            //to shuffle data
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1
#define numTrainingSets 4

int main(void) {

    int trainingSetOrder[] = {0, 1, 2, 3};
    int numberOfEpochs = 10000;

    const double learningRate = 0.1f;
    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutputs];
    double hiddenLayerBias[numHiddenNodes];
    double outputLayerBias[numOutputs];

    double hiddenWeights[numInputs][numHiddenNodes];
    double outputWeights[numHiddenNodes][numOutputs];

    double trainingInputs[numTrainingSets][numInputs] = {{0.0f, 0.0f},
                                                        {1.0f, 0.0f},
                                                        {0.0f, 1.0f},
                                                        {1.0f, 1.0f}};

//0 in last for the XOR
double trainingOutputs[numTrainingSets][numOutputs] = {{0.0f},
                                                        {1.0f},
                                                        {1.0f},
                                                        {0.0f}};

//init weight
for(int i = 0; i < numInputs; i++) {
    for(int j = 0; j < numHiddenNodes; j++){
        hiddenWeights[i][j] = initWeights();
    }
}

for(int i = 0; i < numHiddenNodes; i++) {
    for(int j = 0; j < numOutputs; j++){
        outputWeights[i][j] = initWeights();
    }
}

for(int i = 0; i < numOutputs; i++){
    outputLayerBias[i] = initWeights();
}

// Train the neural network for a number of epochs
for(int epoch = 0; epoch < numberOfEpochs; epoch++){
    shuffle(trainingSetOrder, numTrainingSets);
    for(int x = 0; x < numTrainingSets; x++){
        int i = trainingSetOrder[x];

        //Forward pass
        //Compute hidden layer activation
        for(int j = 0; j < numHiddenNodes; j++){
            double activation = hiddenLayerBias[j];

            for(int k = 0; k < numInputs; k++){
                activation += trainingInputs[i][k] * hiddenWeights[k][j];
            }
            //activation
            hiddenLayer[j] = sigmoid(activation);
        }

        //Compute output layer activation
            for(int j = 0; j < numOutputs; j++){
            double activation = outputLayerBias[j];

            for(int k = 0; k < numHiddenNodes; k++){
                activation += hiddenLayer[k] * outputWeights[k][j];
            }
            //activation
            outputLayer[j] = sigmoid(activation);
        }

        printf("Input: %g   Output: %g   Predicted Output: %g \n",
                trainingInputs[i][0], trainingInputs[i][1],
                trainingOutputs[i][0], outputLayer[0]);

        // Backpropagation
        // Compute change in output weights
        double deltaOutput[numOutputs];

        for(int j = 0; j < numOutputs; j++) {
            double error = trainingOutputs[i][j] - outputLayer[j];
            deltaOutput[j] = error * dSigmoid(outputLayer[j]);
        }

        //Compute change in hidden weights
        double deltaHidden[numHiddenNodes];
        for(int j = 0; j < numHiddenNodes; j++) {
            double error = 0.0f;
            for(int k = 0; k < numOutputs; k++){
                error += deltaOutput[k] * outputWeights[j][k];
            }
            deltaHidden[j] = error * dSigmoid(hiddenLayer[j]);
            }

            //Apply change in output weights
            for(int j = 0; j < numOutputs; j++){
                outputLayerBias[j] += deltaOutput[j] * learningRate;
                for(int k = 0; k < numHiddenNodes; k++){
                    outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * learningRate;

                }
            }

            //Apply change in hidden weights
            for(int j = 0; j < numHiddenNodes; j++){
                hiddenLayerBias[j] += deltaHidden[j] * learningRate;
                for(int k = 0; k < numInputs; k++){
                    hiddenWeights[k][j] += trainingInputs[i][k] * deltaHidden[j] * learningRate;

                }
            }
        }
    }
            //Print final weights after training
            fputs("Final Hidden Weights\n[", stdout);
            for(int j = 0; j < numHiddenNodes; j++){
                fputs("[ ", stdout);
                for(int k =0; k < numInputs; k++){
                    printf("%f", hiddenWeights[k][j]);
                }
                fputs("] ", stdout);
            }


            fputs("]\nFinal Hidden Biases\n", stdout);
            for(int j = 0; j < numHiddenNodes; j++){
                printf("%f", hiddenLayerBias[j]);
            }
            fputs("]\n", stdout);

            fputs("]\nFinal Output Biases\n", stdout);
            for(int j = 0; j < numOutputs; j++){
                printf("%f", outputLayerBias[j]);
            }
            fputs("]\n", stdout);

            return 0;
}



