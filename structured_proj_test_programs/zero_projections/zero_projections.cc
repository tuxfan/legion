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
#include <unistd.h>  // for sleep and usleep
#include "legion.h"

#include "default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;
using namespace LegionRuntime::Accessor;
//using namespace LegionRuntime::Arrays;

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
  COMPUTE_TASK_ANGLE_ID,
  COMPUTE_TASK_AXIS_ALIGNED_ID,
  CHECK_TASK_ID,
};

enum FieldIDs {
  FID_X,
  FID_Y,
  FID_VAL,
};

struct RectDims {
  int side_length_x;
  int side_length_y;
};

struct ComputeArgs {
  int angle;
  int parallel_length;
  int sleep_ms;
};

// A new mapper to control how the index space is sliced
// for testing out performance and correctness with interslice
// dependencies
class SliceMapper : public DefaultMapper {
public:
  SliceMapper(Machine machine, Runtime *rt, Processor local);
public:
  virtual void slice_task(const MapperContext ctx,
                          const Task& task,
                          const SliceTaskInput& input,
                                SliceTaskOutput& output);
};

// pass arguments through to Default
SliceMapper::SliceMapper(Machine m, Runtime *rt, Processor p)
  : DefaultMapper(rt->get_mapper_runtime(), m, p)
{
}

void SliceMapper::slice_task(const MapperContext      ctx,
                             const Task&              task,
                             const SliceTaskInput&    input,
                                   SliceTaskOutput&   output)
{
  // Iterate over all the points and send them all over the world
  output.slices.resize(input.domain.get_volume());
  unsigned idx = 0;

  Machine::ProcessorQuery all_procs(machine);
  all_procs.only_kind(local_cpus[0].kind());
  std::vector<Processor> procs(all_procs.begin(), all_procs.end());

  switch (input.domain.get_dim())
  {
    case 1:
      {
        Rect<1> rect = input.domain;
        for (PointInRectIterator<1> pir(rect);
              pir(); pir++, idx++)
        {
          Rect<1> slice(*pir, *pir);
          output.slices[idx] = TaskSlice(slice,
              procs[idx % procs.size()],
              false/*recurse*/, true/*stealable*/);
        }
        break;
      }
    case 2:
      {
        Rect<2> rect = input.domain;
        for (PointInRectIterator<2> pir(rect);
              pir(); pir++, idx++)
        {
          Rect<2> slice(*pir, *pir);
          output.slices[idx] = TaskSlice(slice,
              procs[idx % procs.size()],
              false/*recurse*/, true/*stealable*/);
        }
        break;
      }
    case 3:
      {
        Rect<3> rect = input.domain;
        for (PointInRectIterator<3> pir(rect);
              pir(); pir++, idx++)
        {
          Rect<3> slice(*pir, *pir);
          output.slices[idx] = TaskSlice(slice,
              procs[idx % procs.size()],
              false/*recurse*/, true/*stealable*/);
        }
        break;
      }
    default:
      assert(false);
  }
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int side_length_x = 4;
  int side_length_y = 4;
  int num_iterations = 1;
  int num_subregions_x = 4; // Assumed to divide side_length_x
  int num_subregions_y = 4; // Assumed to divide side_length_y
  int angle = 225; // angle is measured ccw from positive x-axis
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
      if (!strcmp(command_args.argv[i],"-a"))
        angle = atoi(command_args.argv[++i]);
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

  // Currently only support 2 different angles.
  if (angle != 180 && angle !=  225 && angle != 270)
  {
    printf("Angle must be one of 180 or 270\n");
    assert(0);
  }

  printf("Running computation for (%d, %d) dimensions at angle %d...\n",
      side_length_x, side_length_y, angle);
  printf("Partitioning data into (%d, %d) sub-regions...\n",
      num_subregions_x, num_subregions_y);

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

  int parallel_size = 0;
  // these may be needed later num_waves, subregions_per_wave, perp_size;
  if (angle == 180)
  {
    //perp_size = side_length_x;
    parallel_size = side_length_y;
    //num_waves = num_subregions_x;
    //subregions_per_wave = num_subregions_y;
  }
  else if (angle == 270)
  {
    //perp_size = side_length_y;
    parallel_size = side_length_x;
    //num_waves = num_subregions_y;
    //subregions_per_wave = num_subregions_x;
  }

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
  //Rect<2> launch_bounds(Point<2>(0,0),
      //Point<2>(num_subregions_x-1, num_subregions_y-1));
  //Domain launch_domain = Domain(launch_bounds);
  //ArgumentMap arg_map;

  // First initialize the 'FID_X' and 'FID_Y' fields with some data
  std::vector<Future> init_futures;
  for (int x = num_subregions_x - 1; x >= 0; --x)
  {
    for (int y = num_subregions_y - 1; y >= 0; --y)
    {
      const Point<2> cur_point = Point<2>(x,y);
      LogicalRegion cur_region =
          runtime->get_logical_subregion_by_color(grid_lp, cur_point);

      TaskLauncher init_launcher(INIT_FIELD_TASK_ID,
                                  TaskArgument(NULL, 0));
      init_launcher.add_region_requirement(
          RegionRequirement(cur_region,
                            WRITE_DISCARD, EXCLUSIVE, top_lr));
      init_launcher.add_field(0, FID_X);
      init_launcher.add_field(0, FID_Y);
      init_launcher.add_field(0, FID_VAL);
      init_futures.push_back(runtime->execute_task(ctx, init_launcher));
    }
  }

  for (unsigned i = 0; i < init_futures.size(); i++)
  {
    init_futures[i].get_void_result();
  }
  // Now we launch the computation to calculate Pascal's triangle
  Future f;
  std::vector<Future> parallel_futures;
  double ts_start = Realm::Clock::current_time_in_microseconds();
  for (int j = 0; j < num_iterations; j++)
  {
    for (int x = num_subregions_x - 1; x >= 0; --x)
    {
      for (int y = num_subregions_y - 1; y >= 0; --y)
      {
        const Point<2> cur_point = Point<2>(x,y);
        const Point<2> onex = cur_point + Point<2>(1,0);
        const Point<2> oney = cur_point + Point<2>(0,1);
        LogicalRegion cur_region =
            runtime->get_logical_subregion_by_color(grid_lp, cur_point);
        if (angle == 225)
        {
          ComputeArgs compute_args;
          compute_args.angle = angle; // unused, but might as well be correct
          compute_args.parallel_length = 0; //unused
          compute_args.sleep_ms = sleep_ms;
          LogicalRegion x_region =
              runtime->get_logical_subregion_by_color(grid_lp, onex);
          LogicalRegion y_region =
              runtime->get_logical_subregion_by_color(grid_lp, oney);

          TaskLauncher compute_launcher(COMPUTE_TASK_ANGLE_ID,
              TaskArgument(&compute_args, sizeof(ComputeArgs)));
          compute_launcher.add_region_requirement(
              RegionRequirement(x_region, READ_ONLY, EXCLUSIVE, top_lr));
          compute_launcher.add_region_requirement(
              RegionRequirement(y_region, READ_ONLY, EXCLUSIVE, top_lr));
          compute_launcher.add_region_requirement(
              RegionRequirement(cur_region, READ_WRITE, EXCLUSIVE, top_lr));
          compute_launcher.add_field(0, FID_VAL);
          compute_launcher.add_field(1, FID_VAL);
          compute_launcher.add_field(2, FID_VAL);
          f = runtime->execute_task(ctx, compute_launcher);
        }
        else
        {
          LogicalRegion dep_region;
          if (angle == 180)
          {
            dep_region =
                runtime->get_logical_subregion_by_color(grid_lp, onex);
          }
          else
          {
            dep_region =
                runtime->get_logical_subregion_by_color(grid_lp, oney);
          }
          ComputeArgs compute_args;
          compute_args.angle = angle;
          compute_args.parallel_length = parallel_size;
          compute_args.sleep_ms = sleep_ms;
          TaskLauncher compute_launcher(COMPUTE_TASK_AXIS_ALIGNED_ID,
              TaskArgument(&compute_args, sizeof(ComputeArgs)));
          compute_launcher.add_region_requirement(
              RegionRequirement(dep_region, READ_ONLY, EXCLUSIVE, top_lr));
          compute_launcher.add_region_requirement(
              RegionRequirement(cur_region, READ_WRITE, EXCLUSIVE, top_lr));
          compute_launcher.add_field(0, FID_VAL);
          compute_launcher.add_field(1, FID_VAL);
          parallel_futures.push_back(runtime->execute_task(ctx, compute_launcher));
        }
      }
    }
    if (angle == 225)
    {
      f.get_void_result();
    }
    else
    {
      for (unsigned f_idx = 0; f_idx < parallel_futures.size(); f_idx++)
      {
        parallel_futures[f_idx].get_void_result();
      }
      parallel_futures.clear();
    }
  }

  if (angle == 225)
  {
    f.get_void_result();
  }
  else
  {
    // not running this second loop for now for timing purposes
    // it is already checked above
    for (unsigned f_idx = 0; f_idx < parallel_futures.size() && false; f_idx++)
    {
      parallel_futures[f_idx].get_void_result();
    }
    parallel_futures.clear();
  }

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
  //const int pointx = task->index_point.point_data[0];
  //const int pointy = task->index_point.point_data[1];
  //printf("Initializing fields %d and %d for block (%d, %d) "
      //"with region id %d...\n",
      //fidx, fidy, pointx, pointy,
      //task->regions[0].region.get_index_space().get_id());

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

// Compute the value for each point in the rectangle, takes 3 regions
// for an angular sweep
void compute_task_angle(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  /* UNCOMMENT BELOW FOR DEBUG PRINT STATEMENTS

  printf("Starting the compute task.\n");
  const int pointx = task->index_point.point_data[0];
  const int pointy = task->index_point.point_data[1];
  printf("At point (%d, %d).  My region is %d.  X Region is %d.  "
    "Y Region is %d.\n",
    pointx, pointy,
    task->regions[2].region.get_index_space().get_id(),
    task->regions[0].region.get_index_space().get_id(),
    task->regions[1].region.get_index_space().get_id());*/
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  assert(task->regions[0].privilege_fields.size() == 1);
  assert(task->regions[1].privilege_fields.size() == 1);
  assert(task->regions[2].privilege_fields.size() == 1);

  // Sleep the specified amount
  ComputeArgs compute_args = *((ComputeArgs *)task->args);
  //printf(" I am sleeping for base %d ms\n", compute_args.sleep_ms);
  //printf(" I am sleeping for base %d ms, multiplier %f, for a final value of %d\n", compute_args.sleep_ms, compute_args.sleep_multiplier, compute_args.sleep_ms * compute_args.sleep_multiplier);
  usleep(compute_args.sleep_ms);
  //usleep(compute_args.sleep_ms * 1000);
  
  FieldID val_fid_x_diff = *(task->regions[0].privilege_fields.begin());
  FieldID val_fid_y_diff = *(task->regions[1].privilege_fields.begin());
  FieldID val_fid_curr = *(task->regions[2].privilege_fields.begin());

  RegionAccessor<AccessorType::Generic, int> x_diff_acc = 
    regions[0].get_field_accessor(val_fid_x_diff).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> y_diff_acc = 
    regions[1].get_field_accessor(val_fid_y_diff).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> curr_acc = 
    regions[2].get_field_accessor(val_fid_curr).typeify<int>();

  Rect<2> rect = runtime->get_index_space_domain(ctx,
      task->regions[2].region.get_index_space());

  Domain x_dom = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  Domain y_dom = runtime->get_index_space_domain(ctx,
      task->regions[1].region.get_index_space());
  size_t x_volume = x_dom.get_volume();
  size_t y_volume = y_dom.get_volume();

  Point<2> lo = rect.lo;
  Point<2> hi = rect.hi;
  Point<2> cur_point;
  //printf("starting the compute task for point (%lld, %lld)\n", hi[0], hi[1]);
  int x_diff_val, y_diff_val;
  const Point<2> onex = Point<2>(1,0);
  const Point<2> oney = Point<2>(0,1);

  for (long int x = hi[0]; x >= lo[0]; x--)
  {
    for (long int y = hi[1]; y >= lo[1]; y--)
    {
      cur_point = Point<2>(x, y);
      const Point<2> idx_x = cur_point + onex;
      const Point<2> idx_y = cur_point + oney;
      if (x == hi[0])
      {
        if (x_volume > 0)
        {
          x_diff_val = x_diff_acc.read(idx_x);
        }
        else
        {
          x_diff_val = 0;
        }
      }
      else
      {
        x_diff_val = curr_acc.read(idx_x);
      }
      if (y == hi[1])
      {
        if (y_volume > 0)
        {
          y_diff_val = y_diff_acc.read(idx_y);
        }
        else
        {
          y_diff_val = 0;
        }
      }
      else
      {
        y_diff_val = curr_acc.read(idx_y);
      }
      int computed_val = 0;
      if (x_diff_val > y_diff_val)
      {
        computed_val = x_diff_val + 1;
      }
      else
      {
        computed_val = y_diff_val + 1;
      }
      //printf("x diff is %d, y diff is %d\n", x_diff_val, y_diff_val);
      curr_acc.write(cur_point, computed_val);
    }
  }
}

// Compute the value for each point in the rectangle, takes 2 regions
// for a sweep along one of the major axes.
void compute_task_axis_aligned(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime)
{
  /* UNCOMMENT BELOW FOR DEBUG PRINT STATEMENTS

  printf("Starting the compute task.\n");
  const int pointx = task->index_point.point_data[0];
  const int pointy = task->index_point.point_data[1];
  printf("At point (%d, %d).  My region is %d.  "
    "Other Region is %d.\n",
    pointx, pointy,
    task->regions[1].region.get_index_space().get_id(),
    task->regions[0].region.get_index_space().get_id());*/
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->regions[0].privilege_fields.size() == 1);
  assert(task->regions[1].privilege_fields.size() == 1);
  ComputeArgs compute_args = *((ComputeArgs *)task->args);
  const int angle = compute_args.angle;
  const int parallel_length  = compute_args.parallel_length;

  usleep(compute_args.sleep_ms);

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
    int expected = 1;
    expected = x + y + 1;

    //printf("At point (%lld, %lld)\n", (*pir)[0], (*pir)[1]);
    //printf("Checking for values %d and %d... expected %d, found %d\n",
        //x, y, expected, val);
    
    if (expected != val)
    {
      all_passed = false;
      break;
    }
  }
  if (all_passed)
    printf("SUCCESS!\n");
  else
    printf("FAILURE!\n");
}

void mapper_registration(Machine machine, Runtime *rt,
                          const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    rt->replace_default_mapper(
        new SliceMapper(machine, rt, *it), *it);
  }
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
      TaskConfigOptions(), "top level task");
  Runtime::register_legion_task<init_field_task>(INIT_FIELD_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/, AUTO_GENERATE_ID,
      TaskConfigOptions(), "init task");
  Runtime::register_legion_task<compute_task_angle>(COMPUTE_TASK_ANGLE_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/, AUTO_GENERATE_ID,
      TaskConfigOptions(true,false,false), "compute angle task");
  Runtime::register_legion_task<compute_task_axis_aligned>(
      COMPUTE_TASK_AXIS_ALIGNED_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/, AUTO_GENERATE_ID,
      TaskConfigOptions(true,false,false), "compute axis aligned task");
  Runtime::register_legion_task<check_task>(CHECK_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/, AUTO_GENERATE_ID,
      TaskConfigOptions(), "check task");

  // If using the slicing mapper, add it's callback
  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i],"-sm"))
      Runtime::add_registration_callback(mapper_registration);
  }

  return Runtime::start(argc, argv);
}
