/* Copyright 2017 Stanford University, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "legion.h"
#include "legion_visualization.h"

#include <math.h>

using namespace Legion::Visualization;

enum TaskIDs {
    TOP_LEVEL_TASK_ID,
    GENERATE_IMAGE_DATA_TASK_ID,
    VERIFY_COMPOSITED_IMAGE_DATA_TASK_ID,
};

const int depthFuncs[] = {
    GL_NEVER, GL_LESS, GL_EQUAL, GL_LEQUAL, GL_GREATER, GL_NOTEQUAL, GL_GEQUAL, GL_ALWAYS
};

const int blendFuncs[] = {
    GL_ZERO, GL_ONE, GL_SRC_COLOR, GL_ONE_MINUS_SRC_COLOR, GL_DST_COLOR, GL_ONE_MINUS_DST_COLOR, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA, GL_CONSTANT_COLOR, GL_ONE_MINUS_CONSTANT_COLOR, GL_CONSTANT_ALPHA, GL_ONE_MINUS_CONSTANT_ALPHA,
};

const int numDepthFuncs = sizeof(depthFuncs) / sizeof(depthFuncs[0]);
const int numBlendFuncs = sizeof(blendFuncs) / sizeof(blendFuncs[0]);


static char* paintFileName(char *buffer, int taskID) {
    sprintf(buffer, "/tmp/paint.%d", taskID);
    return buffer;
}

static void paintRegion(ImageSize imageSize,
                        PixelField *r,
                        PixelField *g,
                        PixelField *b,
                        PixelField *a,
                        PixelField *z,
                        PixelField *userdata,
                        ByteOffset stride[IMAGE_REDUCTION_DIMENSIONS],
                        int taskID) {
    
    PixelField zValue = taskID % imageSize.depth;
    char fileName[256];
    FILE *outputFile = fopen(paintFileName(fileName, taskID), "wb");
    
    for(int row = 0; row < imageSize.height; ++row) {
        for(int column = 0; column < imageSize.width; ++column) {
            *r = row;
            *g = column;
            *b = taskID;
            *a = taskID;
            *z = zValue;
            *userdata = taskID;
            
            fwrite(r, sizeof(PixelField), 1, outputFile);
            fwrite(g, sizeof(PixelField), 1, outputFile);
            fwrite(b, sizeof(PixelField), 1, outputFile);
            fwrite(a, sizeof(PixelField), 1, outputFile);
            fwrite(z, sizeof(PixelField), 1, outputFile);
            fwrite(userdata, sizeof(PixelField), 1, outputFile);
            
            r += stride[0]; g += stride[0]; b += stride[0]; a += stride[0]; z += stride[0]; userdata += stride[0];
            zValue = (zValue + 1);
            zValue = (zValue >= imageSize.depth) ? 0 : zValue;
        }
    }
    
    fclose(outputFile);
}



static void generate_image_data_task(const Task *task,
                                     const std::vector<PhysicalRegion> &regions,
                                     Context ctx, HighLevelRuntime *runtime) {
    
    UsecTimer render(ImageReduction::describe_task(task) + ":");
    render.start();
    PhysicalRegion image = regions[0];
    ImageSize imageSize = ((ImageSize *)task->args)[0];
    
    PixelField *r, *g, *b, *a, *z, *userdata;
    ByteOffset stride[IMAGE_REDUCTION_DIMENSIONS];
    int layer = task->get_unique_id() % imageSize.depth;
    Point<IMAGE_REDUCTION_DIMENSIONS> point = Point<IMAGE_REDUCTION_DIMENSIONS>::ZEROES();
    point.x[2] = layer;
    ImageReduction::create_image_field_pointers(imageSize, image, point, r, g, b, a, z, userdata, stride);
    int taskID = task->get_unique_id() % imageSize.depth;
    paintRegion(imageSize, r, g, b, a, z, userdata, stride, taskID);
    render.stop();
    cout << render.to_string() << endl;
}

typedef PixelField* Image;

static void verifyImage(ImageSize imageSize, Image expected, PixelField *r, PixelField *g, PixelField *b, PixelField *a, PixelField *z, PixelField *userdata, ByteOffset stride[]) {

//    for(int i = 0; i < imageSize.pixelsPerLayer(); ++i) {
//        cout << i << ")\t" << r[i] << "\t" << g[i] << "\t" << b[i] << "\t" << a[i] << "\t" << z[i] << "\t" << userdata[i] << endl;
//        cout << "\t" << expected[i*6] << "\t" << expected[i*6+1] << "\t" << expected[i*6+2] << "\t" << expected[i*6+3] << "\t" << expected[i*6+4] << "\t" << expected[i*6+5] << endl;
//    }
//    cout << "-----------------------" << endl;
    
    for(int i = 0; i < imageSize.pixelsPerLayer(); ++i) {
        ///
//        cout << i << ")\t" << r[i] << "\t" << g[i] << "\t" << b[i] << "\t" << a[i] << "\t" << z[i] << "\t" << userdata[i] << endl;
//        cout << "\t" << expected[i*6] << "\t" << expected[i*6+1] << "\t" << expected[i*6+2] << "\t" << expected[i*6+3] << "\t" << expected[i*6+4] << "\t" << expected[i*6+5] << endl;
///
        assert(*expected++ == *r); r += stride[0];
        assert(*expected++ == *g); g += stride[0];
        assert(*expected++ == *b); b += stride[0];
        assert(*expected++ == *a); a += stride[0];
        assert(*expected++ == *z); z += stride[0];
        assert(*expected++ == *userdata); userdata += stride[0];
    }
}



static void verify_composited_image_data_task(const Task *task,
                                              const std::vector<PhysicalRegion> &regions,
                                              Context ctx, HighLevelRuntime *runtime) {
    coord_t layer = task->index_point[2];
    if(layer == 0) {
        PhysicalRegion image = regions[0];
        ImageSize imageSize = ((ImageSize *)task->args)[0];
        Image expected = (Image)((char*)task->args + sizeof(imageSize));
        PixelField *r, *g, *b, *a, *z, *userdata;
        ByteOffset stride[IMAGE_REDUCTION_DIMENSIONS];
        Point<IMAGE_REDUCTION_DIMENSIONS> point = Point<IMAGE_REDUCTION_DIMENSIONS>::ZEROES();
        ImageReduction::create_image_field_pointers(imageSize, image, point, r, g, b, a, z, userdata, stride);
        verifyImage(imageSize, expected, r, g, b, a, z, userdata, stride);
    }
}





static Image loadImage(int taskID, ImageSize imageSize) {
    char fileName[256];
    FILE *inputFile = fopen(paintFileName(fileName, taskID), "rb");
    int numFields = 6;
    Image result = new PixelField[imageSize.pixelsPerLayer() * numFields];
    fread(result, sizeof(PixelField), imageSize.pixelsPerLayer() * numFields, inputFile);
    fclose(inputFile);
    return result;
}


static void compositeTwoImages(Image image0, Image image1, ImageSize imageSize, GLenum depthFunc, GLenum blendFuncSource, GLenum blendFuncDestination) {
    // Use the composite functions from the ImageReductionComposite class.
    // These functions are simple enough to be verified by inspection.
    // There seems no point in writing a duplicate set of functions for testing.
    ImageReductionComposite::CompositeFunction *compositeFunction = ImageReductionComposite::compositeFunctionPointer(depthFunc, blendFuncSource, blendFuncDestination);
    PixelField *r0In = image0;
    PixelField *g0In = r0In + 1;
    PixelField *b0In = g0In + 1;
    PixelField *a0In = b0In + 1;
    PixelField *z0In = a0In + 1;
    PixelField *userdata0In = z0In + 1;
    PixelField *rOut = r0In;
    PixelField *gOut = g0In;
    PixelField *bOut = b0In;
    PixelField *aOut = a0In;
    PixelField *zOut = z0In;
    PixelField *userdataOut = userdata0In;
    PixelField *r1In = image1;
    PixelField *g1In = r1In + 1;
    PixelField *b1In = g1In + 1;
    PixelField *a1In = b1In + 1;
    PixelField *z1In = a1In + 1;
    PixelField *userdata1In = z1In + 1;
    int numPixels = imageSize.numPixelsPerFragment() * imageSize.numFragmentsPerLayer;
    compositeFunction(r0In, g0In, b0In, a0In, z0In, userdata0In, r1In, g1In, b1In, a1In, z1In, userdata1In, rOut, gOut, bOut, zOut, aOut, userdataOut, numPixels);
}


static void verifyAccumulatorMatchesResult(ImageReduction &imageReduction, Image expected, ImageSize imageSize) {
    int numFields = 6;
    int totalSize = imageSize.pixelsPerLayer() * numFields;
    imageReduction.launch_task_by_depth(VERIFY_COMPOSITED_IMAGE_DATA_TASK_ID, expected, totalSize);
}


static void verifyTestResult(ImageReduction &imageReduction, int* permutation, ImageSize imageSize, GLenum depthFunc, GLenum blendFuncSource, GLenum blendFuncDestination) {
    
    int order[imageSize.depth];
    for(int i = 0; i < imageSize.depth; ++i) {
        order[i] = (permutation == NULL) ? i : permutation[i];
    }
    
    Image accumulator = loadImage(order[0], imageSize);
    for(int layer = 1; layer < imageSize.depth; ++layer) {
        Image nextImage = loadImage(order[layer], imageSize);
        compositeTwoImages(accumulator, nextImage, imageSize, depthFunc, blendFuncSource, blendFuncDestination);
    }
    
    verifyAccumulatorMatchesResult(imageReduction, accumulator, imageSize);
}



static void paintImages(ImageSize imageSize, Context context, Runtime *runtime, ImageReduction &imageReduction) {
    FutureMap generateFutures = imageReduction.launch_task_by_depth(GENERATE_IMAGE_DATA_TASK_ID);
    generateFutures.wait_all_results();
}


static void prepareTest(ImageReduction &imageReduction, ImageSize imageSize, Context context, Runtime *runtime, GLenum depthFunc,
                        GLenum blendFuncSource, GLenum blendFuncDestination) {
    paintImages(imageSize, context, runtime, imageReduction);
    if(depthFunc != 0) {
        imageReduction.set_depth_func(depthFunc);
    } else if(blendFuncSource != 0 && blendFuncDestination != 0) {
        imageReduction.set_blend_func(blendFuncSource, blendFuncDestination);
    }
}


static void generateRandomPermutation(int permutation[], int size) {
    static long maxRandom = 0;
    if(maxRandom == 0) {
        maxRandom = powf(2.0f, 31.0f) - 1;
    }
    for(int i = 0; i < size; ++i) {
        permutation[i] = (int)(random() / maxRandom * size);
    }
}


static void testAssociative(ImageSize imageSize, Context context, Runtime *runtime, GLenum depthFunc, GLenum blendFuncSource, GLenum blendFuncDestination) {
    ImageReduction imageReduction(imageSize, context, runtime);
    prepareTest(imageReduction, imageSize, context, runtime, depthFunc, blendFuncSource, blendFuncDestination);
    FutureMap futureMap;
    futureMap = imageReduction.reduce_associative_commutative();
    futureMap.wait_all_results();
    verifyTestResult(imageReduction, NULL, imageSize, depthFunc, blendFuncSource, blendFuncDestination);
    int permutation[imageSize.depth];
    generateRandomPermutation(permutation, imageSize.depth);
    futureMap = imageReduction.reduce_associative_noncommutative(permutation);
    futureMap.wait_all_results();
    verifyTestResult(imageReduction, permutation, imageSize, depthFunc, blendFuncSource, blendFuncDestination);
}



static void testNonassociative(ImageSize imageSize, Context context, Runtime *runtime, GLenum depthFunc, GLenum blendFuncSource, GLenum blendFuncDestination) {
    ImageReduction imageReduction(imageSize, context, runtime);
    prepareTest(imageReduction, imageSize, context, runtime, depthFunc, blendFuncSource, blendFuncDestination);
    FutureMap futureMap;
    futureMap = imageReduction.reduce_nonassociative_commutative();
    futureMap.wait_all_results();
    verifyTestResult(imageReduction, NULL, imageSize, depthFunc, blendFuncSource, blendFuncDestination);
    int permutation[imageSize.depth];
    generateRandomPermutation(permutation, imageSize.depth);
    futureMap = imageReduction.reduce_nonassociative_noncommutative(permutation);
    futureMap.wait_all_results();
    verifyTestResult(imageReduction, permutation, imageSize, depthFunc, blendFuncSource, blendFuncDestination);
    
}


void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime) {
    
    {
        // test with multiple fragments per scanline and all reduction operators
        int width = 64;
        int rows = 48;
        int fragmentsPerLayer = rows * 4;
        
        assert(fragmentsPerLayer > rows && width % (fragmentsPerLayer / rows) == 0);
        ImageSize imageSize = { width, rows, 4, fragmentsPerLayer };
        
        for(int i = 0; i < numDepthFuncs; ++i) {
            GLenum depthFunc = depthFuncs[i];
            testAssociative(imageSize, ctx, runtime, depthFunc, 0, 0);
            testNonassociative(imageSize, ctx, runtime, depthFunc, 0, 0);
        }
        
        for(int i = 0; i < numBlendFuncs; ++i) {
            GLenum sourceFunc = blendFuncs[i];
            for(int j = 0; j < numBlendFuncs; ++j) {
                GLenum destinationFunc = blendFuncs[j];
                testAssociative(imageSize, ctx, runtime, 0, sourceFunc, destinationFunc);
                testNonassociative(imageSize, ctx, runtime, 0, sourceFunc, destinationFunc);
            }
        }
    }
    
    {
        // test with small images
        ImageSize imageSize = { 320, 200, 4, 4 };
        testAssociative(imageSize, ctx, runtime, depthFuncs[0], 0, 0);
        testNonassociative(imageSize, ctx, runtime, depthFuncs[0], 0, 0);
    }
    
    {
        // test with large images
        ImageSize imageSize = { 3840, 2160, 4, 4 };
        testAssociative(imageSize, ctx, runtime, depthFuncs[0], 0, 0);
        testNonassociative(imageSize, ctx, runtime, depthFuncs[0], 0, 0);
    }
    
}



int main(int argc, char *argv[]) {
    HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
                                                           Processor::LOC_PROC, true/*single*/, false/*index*/,
                                                           AUTO_GENERATE_ID, TaskConfigOptions(false/*leaf*/), "top_level_task");
    HighLevelRuntime::register_legion_task<generate_image_data_task>(GENERATE_IMAGE_DATA_TASK_ID,
                                                                     Processor::LOC_PROC, false/*single*/, true/*index*/,
                                                                     AUTO_GENERATE_ID, TaskConfigOptions(true/*leaf*/), "generate_image_data_task");
    HighLevelRuntime::register_legion_task<verify_composited_image_data_task>(VERIFY_COMPOSITED_IMAGE_DATA_TASK_ID,
                                                                              Processor::LOC_PROC, false/*single*/, true/*index*/,
                                                                              AUTO_GENERATE_ID, TaskConfigOptions(true/*leaf*/), "verify_composited_image_data_task");
    
    return HighLevelRuntime::start(argc, argv);
}
