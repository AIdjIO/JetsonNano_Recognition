//Includes
//#include <stdio.h>
//include imageNet header for image recognition
#include <jetson-inference/imageNet.h>

//include loadImage header for loading images
#include <jetson-utils/loadImage.h>

//Declaring main() and Parsing the Command Line
//main entry point
int main(int argc, char** argv){
    //command line argument is expected to provide filename
    //so at least to argument are expected (the 1st arg is the program)

    if (argc < 2)
    {
            printf("my-recognition: expected image filename as argument\n");
            printf("exampleusage: ./JetsonNano_Recognition my_image.jpg\n");
            return 0;
    }
    const char* imgFilename = argv[1];

// Loading the image from disk
// variable to store image data
// image data stored  in share CPU/GPU memory so pointers are used to reference the same physocal memory location.
    float* imgCPU = NULL;
    float* imgCUDA = NULL;

// variable to store image dimensions
int imgWidth = 0;
int imgHeight = 0;

//load the image from disk as float4 RGBA (32bits per channel, 128 bits per pixel)
if (!loadImageRGBA(imgFilename, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight)){
    printf("failed to load image '%s'\n", imgFilename);
    return 0;
}

//Loading the image recognition network
//load googleNet image recognition network with TensorRT
imageNet* net = imageNet::Create(imageNet::GOOGLENET);

//check to make sure that the network model loaded properly
if (!net){
    printf("failed to load image recognition network\n");
    return 0;
}

//Classifying the image
//store confidence of classification
float confidence = 0.0;

//classify the image with TesorRT on the GPU (hence we use the CUDA pointer)
//this returns the index of the object class the image was recognised as (or -1 on error)
const int classIndex = net->Classify(imgCUDA, imgWidth, imgHeight, &confidence);

//Interpreting the Results
//if no error occured
if(classIndex >=0){
    //get class name from index
    const char* classDescription = net->GetClassDesc(classIndex);

    //print out the classification results
    printf("image is recognised as '%s' (class #%i) with %f%% confidence\n",
    classDescription, classIndex, confidence * 100.0f); 
}
else {
    //if Classify() returned < 0, an error occured
    printf("failed to classify imagez\n");
}

//free the network's ressources before shuting down
delete net;

//this is the end of the example
return 0;
}
