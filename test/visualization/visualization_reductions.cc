//
//  visualization_reductions.cpp
//  Legion
//
//  Created by Alan Heirich on 5/22/17.
//
//

#include "legion.h"
#include "legion_visualization.h"

using namespace Legion::Visualization;

enum TaskIDs {
    TOP_LEVEL_TASK_ID,
    GENERATE_IMAGE_DATA_TASK_ID,
};

const int depthFuncs[] = {
    GL_NEVER, GL_LESS, GL_EQUAL, GL_LEQUAL, GL_GREATER, GL_NOTEQUAL, GL_GEQUAL, GL_ALWAYS
};

const int blendFuncs[] = {
    GL_ZERO, GL_ONE, GL_SRC_COLOR, GL_ONE_MINUS_SRC_COLOR, GL_DST_COLOR, GL_ONE_MINUS_DST_COLOR, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA, GL_CONSTANT_COLOR, GL_ONE_MINUS_CONSTANT_COLOR, GL_CONSTANT_ALPHA, GL_ONE_MINUS_CONSTANT_ALPHA,
};

const int numDepthFuncs = sizeof(depthFuncs) / sizeof(depthFuncs[0]);
const int numBlendFuncs = sizeof(blendFuncs) / sizeof(blendFuncs[0]);

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
    
    for(int row = 0; row < imageSize.height; ++row) {
#pragma unroll
        for(int column = 0; column < imageSize.width; ++column) {
            *r = taskID;
            *g = taskID;
            *b = taskID;
            *a = taskID;
            *z = zValue;
            *userdata = taskID;
            r += stride[0]; g += stride[0]; b += stride[0]; a += stride[0]; z += stride[0]; userdata += stride[0];
            zValue = (zValue + 1);
            zValue = (zValue >= imageSize.depth) ? 0 : zValue;
        }
    }
}



void generate_image_data_task(const Task *task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx, HighLevelRuntime *runtime) {
    
    UsecTimer render(ImageReduction::describe_task(task) + ":");
    render.start();
    PhysicalRegion image = regions[0];
    ImageSize imageSize = ((ImageSize *)task->args)[0];
    
    PixelField *r, *g, *b, *a, *z, *userdata;
    ByteOffset stride[IMAGE_REDUCTION_DIMENSIONS];
    int layer = task->get_unique_id() % imageSize.depth;
    ImageReduction::create_image_field_pointers(imageSize, image, layer, r, g, b, a, z, userdata, stride);
    paintRegion(imageSize, r, g, b, a, z, userdata, stride, GENERATE_IMAGE_DATA_TASK_ID);
    render.stop();
    cout << render.to_string() << endl;
}


void treeTestOneOperator(ImageSize imageSize, Context context, Runtime *runtime, GLenum depthFunc, GLenum blendFuncSource, GLenum blendFuncDestination) {
    ImageReduction imageReduction(imageSize, context, runtime);
    if(depthFunc != 0) {
        imageReduction.set_depth_func(depthFunc);
    } else if(blendFuncSource != 0 && blendFuncDestination != 0) {
        imageReduction.set_blend_func(blendFuncSource, blendFuncDestination);
    }
    FutureMap generateFutures = imageReduction.launch_task_by_depth(GENERATE_IMAGE_DATA_TASK_ID);
    generateFutures.wait_all_results();
    
}


void treeTest(ImageSize imageSize, Context context, Runtime *runtime) {
    for(int i = 0; i < numDepthFuncs; ++i) {
        GLenum depthFunc = depthFuncs[i];
        treeTestOneOperator(imageSize, context, runtime, depthFunc, 0, 0);
    }

}


void pipelineTest(ImageSize imageSize, Context context, Runtime *runtime) {
    
}


void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime) {
    
    {
        // test with multiple fragments per scanline
        int width = 64;
        int rows = 48;
        int fragmentsPerLayer = rows * 4;
        assert(fragmentsPerLayer > rows && width % (fragmentsPerLayer / rows) == 0);
        ImageSize imageSize = { width, rows, 4, fragmentsPerLayer };
        treeTest(imageSize, ctx, runtime);
        pipelineTest(imageSize, ctx, runtime);
    }
    
    {
        // test with small images
        ImageSize imageSize = { 320, 200, 4, 4 };
        treeTest(imageSize, ctx, runtime);
        pipelineTest(imageSize, ctx, runtime);
    }
    
    {
        // test with large images
        ImageSize imageSize = { 3840, 2160, 4, 4 };
        treeTest(imageSize, ctx, runtime);
        pipelineTest(imageSize, ctx, runtime);
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
    return HighLevelRuntime::start(argc, argv);
}
