/* Copyright 2016 Stanford University
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


#include <cstdio>
#include <cassert>
#include <cstdlib>
#include "legion.h"
#include <unistd.h>  // for sleep and usleep
using namespace Legion;
using namespace LegionRuntime::Accessor;

/*
 * In this example we illustrate how the Legion
 * programming model supports multiple partitions
 * of the same logical region and the benefits it
 * provides by allowing multiple views onto the
 * same logical region.  We compute a simple 5-point
 * 1D stencil using the standard formula:
 * f'(x) = (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h))/12h
 * For simplicity we'll assume h=1.
 */

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_FIELD_TASK_ID,
  COMPUTE_TASK_ID,
  CHECK_TASK_ID,
  PAUSE_TASK_ID,
};

enum FieldIDs {
  FID_X,
  FID_Y,
  FID_VAL,
};

enum ProjIDs {
  X_PROJ = 1,
  Y_PROJ = 2,
  ID_PROJ = 3,
};

struct RectDims {
  int side_length_x;
  int side_length_y;
};

struct ComputeArgs {
  int sleep_ms;
  int parallel_length;
  int wave_num;
};

class IDProjectionFunctor : public ProjectionFunctor
{
  public:
    IDProjectionFunctor()
      : ProjectionFunctor() {}

    ~IDProjectionFunctor() {}

    virtual LogicalRegion project(Context ctx, Task *task, unsigned index,
                                  LogicalRegion upper_bound,
                                  const DomainPoint &point)
    {
      assert(0);
    }

    virtual LogicalRegion project(Context ctx, Task *task, unsigned index,
                                  LogicalPartition upper_bound,
                                  const DomainPoint &point)
    {
      ComputeArgs compute_args = *((ComputeArgs *)task->args);
      const int point_in_launch = point[0];
      const int wave_num = compute_args.wave_num;
      
      Point<2> proj_point(wave_num, point_in_launch);

      LogicalRegion ret_region =
          runtime->get_logical_subregion_by_color(ctx, upper_bound, DomainPoint(proj_point));
      return ret_region;
    }

    virtual unsigned get_depth() const {
      return 0;
    }
};


//class SingleDiffProjectionFunctor : public StructuredProjectionFunctor
class XDiffProjectionFunctor : public ProjectionFunctor
{
  public:
    XDiffProjectionFunctor()
      : ProjectionFunctor() {}

    ~XDiffProjectionFunctor() {}

    virtual LogicalRegion project(Context ctx, Task *task, unsigned index,
                                  LogicalRegion upper_bound,
                                  const DomainPoint &point)
    {
      assert(0);
    }

    virtual LogicalRegion project(Context ctx, Task *task, unsigned index,
                                  LogicalPartition upper_bound,
                                  const DomainPoint &point)
    {
      ComputeArgs compute_args = *((ComputeArgs *)task->args);
      const int point_in_launch = point[0];
      const int wave_num = compute_args.wave_num;
      
      Point<2> proj_point(wave_num + 1, point_in_launch);

      LogicalRegion ret_region =
          runtime->get_logical_subregion_by_color(ctx, upper_bound, DomainPoint(proj_point));
      return ret_region;
    }

    virtual unsigned get_depth() const {
      return 0;
    }
};


void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int side_length_x = 4;
  int side_length_y = 4;
  int num_iterations = 1;
  int num_subregions_x = 4; // Assumed to divide side_length_x
  int num_subregions_y = 4; // Assumed to divide side_length_y
  int sleep_ms = 0; // how long each task should sleep in milliseconds

  // say it's disjoint by default,
  // give flag for toggling to force it to compute disjointedness
  PartitionKind partition_kind = DISJOINT_KIND;

  // Check for any command line arguments
  {
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-n"))
      {
        side_length_x = atoi(command_args.argv[++i]);
        side_length_y = side_length_x;
      }
      if (!strcmp(command_args.argv[i],"-nx"))
        side_length_x = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-ny"))
        side_length_y = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-b"))
      {
        num_subregions_x = atoi(command_args.argv[++i]);
        num_subregions_y = num_subregions_x;
      }
      if (!strcmp(command_args.argv[i],"-bx"))
        num_subregions_x = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-by"))
        num_subregions_y = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-i"))
        num_iterations = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-c"))
        partition_kind = COMPUTE_KIND;
      if (!strcmp(command_args.argv[i],"-sms"))
        sleep_ms = atoi(command_args.argv[++i]);
    }
  }

  if (side_length_x % num_subregions_x != 0 ||
      side_length_y % num_subregions_y != 0)
  {
    printf("subregions per side must evenly divide side length!\n");
    assert(0);
  }

  printf("Running computation for (%d, %d) dimensions at angle 180...\n",
      side_length_x, side_length_y);
  printf("Partitioning data into (%d, %d) sub-regions...\n", num_subregions_x,
      num_subregions_y);

  // For this example we'll create a single logical region with two
  // fields.  We'll initialize the field identified by 'FID_X' and 'FID_Y' with
  // our input data and then compute the value and write into 'FID_VAL'.
  Rect<2> elem_rect(Point<2>(0,0), Point<2>(side_length_x-1,
      side_length_y-1));
  IndexSpace is = runtime->create_index_space(ctx, elem_rect);
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(int),FID_X);
    allocator.allocate_field(sizeof(int),FID_Y);
    allocator.allocate_field(sizeof(int),FID_VAL);
  }
  LogicalRegion top_lr = runtime->create_logical_region(ctx, is, fs);

/* TODO: come back and delete this!
  int num_waves, parallel_size, subregions_per_wave, perp_size;
  if (angle == 180)
  {
    perp_size = side_length_x;
    parallel_size = side_length_y;
    num_waves = num_subregions_x;
    subregions_per_wave = num_subregions_y;
  }
  else
  {
    perp_size = side_length_y;
    parallel_size = side_length_x;
    num_waves = num_subregions_y;
    subregions_per_wave = num_subregions_x;
  }
  */

  // Make our color_domain based on the number of subregions
  // that we want to create.  We create extra empty subregions
  // around the outside of the grid so that the compute tasks
  // can all work the same.
  Rect<2> color_bounds(Point<2>(0,0),
      Point<2>(num_subregions_x, num_subregions_y));
  Domain color_domain = Domain(color_bounds);

  // Create (possibly coarser) grid partition of the grid of points
  IndexPartition grid_ip;
  {
    const int points_per_partition_x = side_length_x/num_subregions_x;
    const int points_per_partition_y = side_length_y/num_subregions_y;
    DomainPointColoring d_coloring;
    for (Domain::DomainPointIterator itr(color_domain); itr; itr++)
    {
      // Make the empty bounding subregions.
      if (itr.p[0] == num_subregions_x || itr.p[1] == num_subregions_y)
      {
          Rect<2> subrect(Point<2>(0,0),Point<2>(-1,-1));
          d_coloring[itr.p] = Domain(subrect);
          continue;
      }
      int x_start = itr.p[0] * points_per_partition_x;
      int y_start = itr.p[1] * points_per_partition_y;
      int x_end = (itr.p[0] + 1) * points_per_partition_x - 1;
      int y_end = (itr.p[1] + 1) * points_per_partition_y - 1;
      Rect<2> subrect(Point<2>(x_start, y_start),Point<2>(x_end, y_end));
      d_coloring[itr.p] = Domain(subrect);
    }


    // Once we've computed our coloring then we can
    // create our partition.
    grid_ip = runtime->create_index_partition(ctx, is, color_domain,
                                    d_coloring, partition_kind);
  }

  // Once we've created our index partitions, we can get the
  // corresponding logical partitions for the top_lr
  // logical region.
  LogicalPartition grid_lp = 
    runtime->get_logical_partition(ctx, top_lr, grid_ip);

  // Our launch domain will again be only include the data subregions
  // and not the dummy ones
  Rect<2> launch_bounds(Point<2>(0,0),
      Point<2>(num_subregions_x-1, num_subregions_y-1));
  Domain launch_domain = Domain(launch_bounds);
  ArgumentMap arg_map;

  FutureMap fm;
  // First initialize the 'FID_X' and 'FID_Y' fields with some data
  IndexLauncher init_launcher(INIT_FIELD_TASK_ID, launch_domain,
                              TaskArgument(NULL, 0), arg_map);
  init_launcher.add_region_requirement(
      RegionRequirement(grid_lp, 0/*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, top_lr));
  init_launcher.add_field(0, FID_X);
  init_launcher.add_field(0, FID_Y);
  init_launcher.add_field(0, FID_VAL);
  fm = runtime->execute_index_space(ctx, init_launcher);

  fm.wait_all_results();
  double ts_start = Realm::Clock::current_time_in_microseconds();
  for (int j = 0; j < num_iterations; j++)
  {
    for (int i = num_subregions_x - 1; i >= 0; i--)
    {
      ComputeArgs compute_args;
      compute_args.sleep_ms = sleep_ms;
      compute_args.parallel_length = side_length_y;
      compute_args.wave_num = i;

      Rect<1> wave_launch_bounds(Point<1>(0),
          Point<1>(num_subregions_y - 1));
      Domain wave_launch_domain = Domain(wave_launch_bounds);

      IndexLauncher compute_launcher(COMPUTE_TASK_ID, wave_launch_domain,
           TaskArgument(&compute_args, sizeof(ComputeArgs)), arg_map);
      compute_launcher.add_region_requirement(
          RegionRequirement(grid_lp, X_PROJ,
                            READ_ONLY, EXCLUSIVE, top_lr));
      compute_launcher.add_region_requirement(
          RegionRequirement(grid_lp, ID_PROJ,
                            READ_WRITE, EXCLUSIVE, top_lr));
      compute_launcher.add_field(0, FID_VAL);
      compute_launcher.add_field(1, FID_VAL);

      fm = runtime->execute_index_space(ctx, compute_launcher);
    }
    fm.wait_all_results();
  }

  fm.wait_all_results();
  double ts_end = Realm::Clock::current_time_in_microseconds();
  double sim_time = 1e-6 * (ts_end - ts_start);
  fprintf(stderr, "ELAPSED TIME = %7.7f s\n", sim_time);

  // We got what we need, force the run to end
  //assert(0);

  // Finally, we launch a single task to check the results.
  RectDims rect_dims;
  rect_dims.side_length_x = side_length_x;
  rect_dims.side_length_y = side_length_y;
  TaskLauncher check_launcher(CHECK_TASK_ID, 
      TaskArgument(&rect_dims, sizeof(RectDims)));
  check_launcher.add_region_requirement(
      RegionRequirement(top_lr, READ_ONLY, EXCLUSIVE, top_lr));
  check_launcher.add_field(0, FID_X);
  check_launcher.add_field(0, FID_Y);
  check_launcher.add_field(0, FID_VAL);
  //runtime->execute_task(ctx, check_launcher);

  // Clean up our region, index space, and field space
  //runtime->destroy_logical_region(ctx, top_lr);
  //runtime->destroy_field_space(ctx, fs);
  //runtime->destroy_index_space(ctx, is);
}

// The standard initialize field task from earlier examples
void init_field_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 3);

  std::set<unsigned int>::iterator fields =
      task->regions[0].privilege_fields.begin();
  FieldID fidx = *fields;
  FieldID fidy = *(++fields);
  FieldID fid_val_write = *(++fields);

  RegionAccessor<AccessorType::Generic, int> accx = 
    regions[0].get_field_accessor(fidx).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> accy = 
    regions[0].get_field_accessor(fidy).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> acc_val_write = 
    regions[0].get_field_accessor(fid_val_write).typeify<int>();

  Rect<2> rect = runtime->get_index_space_domain(ctx, 
      task->regions[0].region.get_index_space());
  for (PointInRectIterator<2> pir(rect); pir(); pir++)
  {
    accx.write(*pir, (*pir)[0]);
    accy.write(*pir, (*pir)[1]);
    acc_val_write.write(*pir, 1);
  }
}

// Compute the value triangle value for each point in the rectangle
void compute_task(const Task *task,
                  const std::vector<PhysicalRegion> &regions,
                  Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->regions[0].privilege_fields.size() == 1);
  assert(task->regions[1].privilege_fields.size() == 1);
  ComputeArgs compute_args = *((ComputeArgs *)task->args);
  const int angle = 180;
  const int parallel_length  = compute_args.parallel_length;
  usleep(compute_args.sleep_ms);
  //usleep(compute_args.sleep_ms * 1000);

  FieldID val_prev = *(task->regions[0].privilege_fields.begin());
  FieldID val_curr = *(task->regions[1].privilege_fields.begin());

  RegionAccessor<AccessorType::Generic, int> prev_acc =
    regions[0].get_field_accessor(val_prev).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> curr_acc =
    regions[1].get_field_accessor(val_curr).typeify<int>();

  Rect<2> rect = runtime->get_index_space_domain(ctx,
      task->regions[1].region.get_index_space());

  Domain prev_dom = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  size_t prev_volume = prev_dom.get_volume();

  Point<2> lo = rect.lo;
  Point<2> hi = rect.hi;
  Point<2> cur_point;
  int prev_val;
  Point<2> offset_prev_val;
  int primary_loop_index;
  if (angle == 180)
  {
    offset_prev_val = Point<2>(1,0);
    primary_loop_index = 0;
  }
  else
  {
    offset_prev_val = Point<2>(0,1);
    primary_loop_index = 1;
  }

  for (long int i = hi[primary_loop_index]; i >= lo[primary_loop_index]; i--)
  {
    for (long int j = hi[1-primary_loop_index];
        j >= lo[1-primary_loop_index]; j--)
    {
      if (angle == 180)
      {
        cur_point = Point<2>(i, j);
      }
      else
      {
        cur_point = Point<2>(j, i);
      }
      const Point<2> access_idx = cur_point + offset_prev_val;
      if (i == hi[primary_loop_index])
      {
        if (prev_volume > 0)
        {
          prev_val = prev_acc.read(access_idx);
        }
        else
        {
          // make the starting value be the index of the minor axis
          // (parallel to wave direction)
          prev_val = parallel_length - 1 - j;
        }
      }
      else
      {
        prev_val = curr_acc.read(access_idx);
      }
      int computed_val = prev_val + 1;
      curr_acc.write(cur_point, computed_val);
    }
  }
}

void pause_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  unsigned int guess = 0;
  for (int i = 0; i < 10000; i++)
  {
    guess = guess + 1;
  }
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->regions[0].privilege_fields.size() == 1);
  assert(task->regions[1].privilege_fields.size() == 10000/guess);
}

void check_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 3);
  assert(task->arglen == sizeof(RectDims));
  RectDims rect_dims = *((RectDims *)task->args);
  const int side_length_x = rect_dims.side_length_x;
  const int side_length_y = rect_dims.side_length_y;

  std::set<unsigned int>::iterator fields =
      task->regions[0].privilege_fields.begin();
  FieldID fidx = *fields;
  FieldID fidy = *(++fields);
  FieldID fid_val = *(++fields);

  RegionAccessor<AccessorType::Generic, int> accx = 
    regions[0].get_field_accessor(fidx).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> accy = 
    regions[0].get_field_accessor(fidy).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> acc_val = 
    regions[0].get_field_accessor(fid_val).typeify<int>();

  Rect<2> rect = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());

  // This is the checking task so we can just do the slow path
  bool all_passed = true;
  for (PointInRectIterator<2> pir(rect); pir(); pir++)
  {
    int x = side_length_x - 1 - accx.read(*pir);
    int y = side_length_y - 1 - accy.read(*pir);
    int val = acc_val.read(*pir);
    int expected = x + y + 1;

    //printf("At point (%lld, %lld)\n", (*pir)[0], (*pir)[1]);
    //printf("Checking for values %d and %d... expected %d, found %d\n",
        //x, y, expected, val);
    
    if (expected != val)
    {
      all_passed = false;
      //break;
    }
  }
  if (all_passed)
    printf("SUCCESS!\n");
  else
    printf("FAILURE!\n");
}

int main(int argc, char **argv)
{
  Processor::Kind top_level_proc = Processor::LOC_PROC;
  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i],"-ll:io"))
    {
      int io_procs = atoi(argv[++i]);
      if (io_procs >= 1)
        top_level_proc = Processor::IO_PROC;
    }
  }

  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  Runtime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
      top_level_proc, true/*single*/, false/*index*/, AUTO_GENERATE_ID,
      TaskConfigOptions(), "top_level_task");
  Runtime::register_legion_task<init_field_task>(INIT_FIELD_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/, AUTO_GENERATE_ID,
      TaskConfigOptions(), "init_task");
  Runtime::register_legion_task<compute_task>(COMPUTE_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/, AUTO_GENERATE_ID,
      TaskConfigOptions(true, false, false), "compute_task");
  Runtime::register_legion_task<pause_task>(PAUSE_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/, AUTO_GENERATE_ID,
      TaskConfigOptions(true, false, false), "pause_task");
  Runtime::register_legion_task<check_task>(CHECK_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/, AUTO_GENERATE_ID,
      TaskConfigOptions(), "check_task");

  Runtime::preregister_projection_functor(X_PROJ, new XDiffProjectionFunctor());
  Runtime::preregister_projection_functor(ID_PROJ, new IDProjectionFunctor());

  return Runtime::start(argc, argv);
}
