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

namespace Legion {
  namespace Visualization {
    
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
                            Legion::ByteOffset stride[IMAGE_REDUCTION_DIMENSIONS],
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
      ImageReduction::create_image_field_pointers(imageSize, image, r, g, b, a, z, userdata, stride, runtime, ctx);
      int taskID = task->get_unique_id() % imageSize.depth;
      paintRegion(imageSize, r, g, b, a, z, userdata, stride, taskID);
      render.stop();
      cout << render.to_string() << endl;
    }
    
    typedef PixelField* Image;
    
    static bool verifyPixelField(int pixelID, char *fieldName, PixelField expected, PixelField actual) {
      if(expected != actual) {
        std::cerr << "verification failue at pixel " << pixelID << " field " << fieldName << " expected " << expected << " saw " << actual << std::endl;
        return false;
      }
      return true;
    }
    
    static void verifyImage(ImageSize imageSize, Image expected, PixelField *r, PixelField *g, PixelField *b, PixelField *a, PixelField *z, PixelField *userdata, ByteOffset stride[]) {
      
      for(int i = 0; i < imageSize.pixelsPerLayer(); ++i) {
        assert(verifyPixelField(i, (char*)"r", *expected++, *r)); r += stride[0];
        assert(verifyPixelField(i, (char*)"g", *expected++, *g)); g += stride[0];
        assert(verifyPixelField(i, (char*)"b", *expected++, *b)); b += stride[0];
        assert(verifyPixelField(i, (char*)"a", *expected++, *a)); a += stride[0];
        assert(verifyPixelField(i, (char*)"z", *expected++, *z)); z += stride[0];
        assert(verifyPixelField(i, (char*)"userdata", *expected++, *userdata)); userdata += stride[0];
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
        ImageReduction::create_image_field_pointers(imageSize, image, r, g, b, a, z, userdata, stride, runtime, ctx);
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
    
    
    
    ///debugging
    static void dumpImage(PixelField *image, char *text) {
      std::cout << std::endl;
      std::cout << text << std::endl;
      PixelField *foo = image;
      for(int i = 0; i < 10; ++i) {
        std::cout << text << " pixel " << i << ": ";
        std::cout << foo[0] << "\t" << foo[1] << "\t" << foo[2] << "\t" << foo[3] << "\t" << foo[4] << "\t" << foo[5] << std::endl;
      }
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
      
//      dumpImage(image0, "image 0"); dumpImage(image1, "image 1");
      
      int numPixels = imageSize.numPixelsPerFragment() * imageSize.numFragmentsPerLayer;
      compositeFunction(r0In, g0In, b0In, a0In, z0In, userdata0In, r1In, g1In, b1In, a1In, z1In, userdata1In, rOut, gOut, bOut, zOut, aOut, userdataOut, numPixels);
      
//      dumpImage(image0, "result");
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
    
    
    
    static void generateOrdering(int ordering[], int numElements) {
      
    }
    
    
    static void testAssociative(ImageSize imageSize, Context context, Runtime *runtime, GLenum depthFunc, GLenum blendFuncSource, GLenum blendFuncDestination) {
      ImageReduction imageReduction(imageSize, context, runtime);
      prepareTest(imageReduction, imageSize, context, runtime, depthFunc, blendFuncSource, blendFuncDestination);
      FutureMap futureMap;
      futureMap = imageReduction.reduce_associative_commutative();
      futureMap.wait_all_results();
      verifyTestResult(imageReduction, NULL, imageSize, depthFunc, blendFuncSource, blendFuncDestination);
      futureMap = imageReduction.reduce_associative_noncommutative();
      futureMap.wait_all_results();
      int ordering[imageSize.depth];
      generateOrdering(ordering, imageSize.depth);
      verifyTestResult(imageReduction, ordering, imageSize, depthFunc, blendFuncSource, blendFuncDestination);
    }
    
    
    
    static void testNonassociative(ImageSize imageSize, Context context, Runtime *runtime, GLenum depthFunc, GLenum blendFuncSource, GLenum blendFuncDestination) {
      ImageReduction imageReduction(imageSize, context, runtime);
      prepareTest(imageReduction, imageSize, context, runtime, depthFunc, blendFuncSource, blendFuncDestination);
      FutureMap futureMap;
      futureMap = imageReduction.reduce_nonassociative_commutative();
      futureMap.wait_all_results();
      verifyTestResult(imageReduction, NULL, imageSize, depthFunc, blendFuncSource, blendFuncDestination);
      futureMap = imageReduction.reduce_nonassociative_noncommutative();
      futureMap.wait_all_results();
      int ordering[imageSize.depth];
      generateOrdering(ordering, imageSize.depth);
      verifyTestResult(imageReduction, ordering, imageSize, depthFunc, blendFuncSource, blendFuncDestination);
      
    }
    
    const int numDomainNodesX = 2;
    const int numDomainNodesY = 2;
    const int numDomainNodesZ = 1;
    const int numDomainNodes = numDomainNodesX * numDomainNodesY * numDomainNodesZ;
    
    
    void top_level_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, HighLevelRuntime *runtime) {
      
      {
        // test with multiple fragments per scanline and all reduction operators
        const int width = 64;
        const int rows = 48;
//        const int fragmentsPerLayer = rows * 4;
        
//        assert(fragmentsPerLayer > rows && width % (fragmentsPerLayer / rows) == 0);
        
        const int fragmentsPerLayer = 1;//testing
        
        ImageSize imageSize = { width, rows, numDomainNodes, fragmentsPerLayer };
        
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
      
      const int fewFragmentsPerLayer = 4;

      {
        // test with small images
        ImageSize imageSize = { 320, 200, numDomainNodes, fewFragmentsPerLayer };
        testAssociative(imageSize, ctx, runtime, depthFuncs[0], 0, 0);
        testNonassociative(imageSize, ctx, runtime, depthFuncs[0], 0, 0);
      }
      
      {
        // test with large images
        ImageSize imageSize = { 3840, 2160, numDomainNodes, fewFragmentsPerLayer };
        testAssociative(imageSize, ctx, runtime, depthFuncs[0], 0, 0);
        testNonassociative(imageSize, ctx, runtime, depthFuncs[0], 0, 0);
      }
      
    }
  }
}


static void preregisterSimulationBounds(int numSimulationBoundsX, int numSimulationBoundsY, int numSimulationBoundsZ) {

  // call this before starting the Legion runtime
  
  int numDomainNodes = numDomainNodesX * numDomainNodesY * numDomainNodesZ;
  const int fieldsPerBound = 6;
  float *bounds = new float[numDomainNodes * fieldsPerBound];
  float *boundsPtr = bounds;
  
  // construct a simple regular simulation domain
  for(int x = 0; x < numDomainNodesX; ++x) {
    for(int y = 0; y < numDomainNodesY; ++y) {
      for(int z = 0; z < numDomainNodesZ; ++z) {
        *boundsPtr++ = x;
        *boundsPtr++ = y;
        *boundsPtr++ = z;
        *boundsPtr++ = x + 1;
        *boundsPtr++ = y + 1;
        *boundsPtr++ = z + 1;
      }
    }
  }
  
  int numSimulationBounds = numDomainNodesX * numDomainNodesY * numDomainNodesZ;
  Legion::Visualization::ImageReduction::preregisterSimulationBounds(bounds, numSimulationBounds);
  delete [] bounds;
}



int main(int argc, char *argv[]) {
  
  preregisterSimulationBounds(numDomainNodesX, numDomainNodesY, numDomainNodesZ);
  
  Legion::HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  Legion::HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
                                                                 Legion::Processor::LOC_PROC, true/*single*/, false/*index*/,
                                                                 AUTO_GENERATE_ID, Legion::TaskConfigOptions(false/*leaf*/), "top_level_task");
  Legion::HighLevelRuntime::register_legion_task<generate_image_data_task>(GENERATE_IMAGE_DATA_TASK_ID,
                                                                           Legion::Processor::LOC_PROC, false/*single*/, true/*index*/,
                                                                           AUTO_GENERATE_ID, Legion::TaskConfigOptions(true/*leaf*/), "generate_image_data_task");
  Legion::HighLevelRuntime::register_legion_task<verify_composited_image_data_task>(VERIFY_COMPOSITED_IMAGE_DATA_TASK_ID,
                                                                                    Legion::Processor::LOC_PROC, false/*single*/, true/*index*/,
                                                                                    AUTO_GENERATE_ID, Legion::TaskConfigOptions(true/*leaf*/), "verify_composited_image_data_task");
  
  return Legion::HighLevelRuntime::start(argc, argv);
}
