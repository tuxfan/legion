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
using namespace Legion;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

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
  LAUNCHER_HELPER_TASK_ID,
  COMPUTE_TASK_ID,
  CHECK_TASK_ID,
  PAUSE_TASK_ID,
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
  int angle = 180; // angle is measured ccw from positive x-axis

  // say it's disjoint by default,
  // give flag for toggling to force it to compute disjointedness
  PartitionKind partition_kind = DISJOINT_KIND;

  // Check for any command line arguments
  {
      const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-n")) {
        side_length_x = 1 << atoi(command_args.argv[++i]);
        side_length_y = side_length_x;
      }
      if (!strcmp(command_args.argv[i],"-nx"))
        side_length_x = 1 << atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-ny"))
        side_length_y = 1 << atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-b")) {
        num_subregions_x = 1 << atoi(command_args.argv[++i]);
        num_subregions_y = num_subregions_x;
      }
      if (!strcmp(command_args.argv[i],"-bx"))
        num_subregions_x = 1 << atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-by"))
        num_subregions_y = 1 << atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-i"))
        num_iterations = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-c"))
        partition_kind = COMPUTE_KIND;
      if (!strcmp(command_args.argv[i],"-a"))
        angle = atoi(command_args.argv[++i]);
    }
  }

  if (side_length_x % num_subregions_x != 0 || side_length_y % num_subregions_y != 0) {
    printf("subregions per side must evenly divide side length!\n");
    assert(0);
  }

  // Currently only support 2 different angles.
  if (angle != 180 && angle != 270) {
    printf("Angle must be one of 180 or 270\n");
    assert(0);
  }

  printf("Running computation for (%d, %d) dimensions at angle %d...\n",
      side_length_x, side_length_y, angle);
  printf("Partitioning data into (%d, %d) sub-regions...\n", num_subregions_x, num_subregions_y);

  // For this example we'll create a single logical region with three
  // fields.  We'll initialize the field identified by 'FID_X' and 'FID_Y' with
  // our input data and then compute the value and write into 'FID_VAL'.
  Rect<2> elem_rect(make_point(0,0),make_point(side_length_x-1, side_length_y-1));
  IndexSpace is = runtime->create_index_space(ctx, 
                          Domain::from_rect<2>(elem_rect));
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(int),FID_X);
    allocator.allocate_field(sizeof(int),FID_Y);
    allocator.allocate_field(sizeof(int),FID_VAL);
  }
  LogicalRegion top_lr = runtime->create_logical_region(ctx, is, fs);

  int num_waves, parallel_size, subregions_per_wave, perp_size;
  if (angle == 180) {
    perp_size = side_length_x;
    parallel_size = side_length_y;
    num_waves = num_subregions_x;
    subregions_per_wave = num_subregions_y;
  }
  else {
    perp_size = side_length_y;
    parallel_size = side_length_x;
    num_waves = num_subregions_y;
    subregions_per_wave = num_subregions_x;
  }
  
  // Make our color_domain based on the number of subregions
  // that we want to create, subtracting 1 for 0 indexing.
  Rect<1> color_bounds(make_point(0), make_point(num_waves));
  Domain color_domain = Domain::from_rect<1>(color_bounds);

  // Create two levels of partition so that we can launch
  // the different wavefronts in parallel.
  IndexPartition first_ip;
  LogicalPartition first_lp;
  {
    DomainPointColoring d_coloring;
    const int points_per_partition_x = side_length_x/num_subregions_x;
    const int points_per_partition_y = side_length_y/num_subregions_y;

    // we include an empty extra partition for the first compute task
    for (int i = 0; i <= num_waves; i++) {
      if (i == num_waves) {
        // Map the last point to the empty domain
        Rect<2> subrect(make_point(0,0),make_point(-1,-1));
        d_coloring[DomainPoint::from_point<1>(make_point(i))] = Domain::from_rect<2>(subrect);
        continue;
      }
      int x_start, x_end, y_start, y_end;
      if (angle == 180) {
        x_start = i * points_per_partition_x;
        y_start = 0;
        x_end = x_start + points_per_partition_x - 1;
        y_end = side_length_y - 1;
      } else {
        x_start = 0;
        y_start = i * points_per_partition_y;
        x_end = side_length_x - 1;
        y_end = y_start + points_per_partition_y - 1;
      }
      Rect<2> subrect(make_point(x_start, y_start),make_point(x_end, y_end));
      d_coloring[DomainPoint::from_point<1>(make_point(i))] = Domain::from_rect<2>(subrect);
    }

    first_ip = runtime->create_index_partition(ctx, is, color_domain, d_coloring, partition_kind);
    first_lp = runtime->get_logical_partition(ctx, top_lr, first_ip);

    for (Domain::DomainPointIterator itr(color_domain); itr; itr++) {
      Rect<1> sub_color_bounds(make_point(0), make_point(subregions_per_wave-1));
      Domain sub_color_domain = Domain::from_rect<1>(sub_color_bounds);
      DomainPointColoring sub_d_coloring;
      LogicalRegion to_partition = runtime->get_logical_subregion_by_color(ctx,
          first_lp, itr.p);

      for (Domain::DomainPointIterator itr2(sub_color_domain); itr2; itr2++) {
        if (itr.p[0] == num_waves) {
          // Further divide the empty region into more empty regions for symmetry
          Rect<2> subrect(make_point(0,0),make_point(-1,-1));
          sub_d_coloring[itr2.p] = Domain::from_rect<2>(subrect);
          continue;
        }
        int x_start, x_end, y_start, y_end;
        if (angle == 180) {
          x_start = itr.p[0] * points_per_partition_x;
          y_start = itr2.p[0] * points_per_partition_y;
        } else {
          x_start = itr2.p[0] * points_per_partition_x;
          y_start = itr.p[0] * points_per_partition_y;
        }
        x_end = x_start + points_per_partition_x - 1;
        y_end = y_start + points_per_partition_y - 1;

        Rect<2> subrect(make_point(x_start, y_start),make_point(x_end, y_end));
        sub_d_coloring[itr2.p] = Domain::from_rect<2>(subrect);
      }

      runtime->create_index_partition(ctx, to_partition.get_index_space(),
          sub_color_domain, sub_d_coloring, partition_kind);
    }
  }

  // Create a full partitioning to do a normal initialization index space launch
  Rect<2> full_color_bounds(make_point(0,0),
      make_point(num_subregions_x - 1, num_subregions_y - 1));
  Domain full_color_domain = Domain::from_rect<2>(full_color_bounds);
  IndexPartition full_ip;
  {
    DomainPointColoring d_coloring;
    const int points_per_partition_x = side_length_x/num_subregions_x;
    const int points_per_partition_y = side_length_y/num_subregions_y;
    for (Domain::DomainPointIterator itr(full_color_domain); itr; itr++) {
      int x_start = itr.p[0] * points_per_partition_x;
      int y_start = itr.p[1] * points_per_partition_y;
      int x_end = x_start + points_per_partition_x - 1;
      int y_end = y_start + points_per_partition_y - 1;
      Rect<2> subrect(make_point(x_start, y_start),make_point(x_end, y_end));
      d_coloring[itr.p] = Domain::from_rect<2>(subrect);
    }
    full_ip = runtime->create_index_partition(ctx, is, full_color_domain, d_coloring, partition_kind);
  }
  LogicalPartition full_lp = runtime->get_logical_partition(ctx, top_lr, full_ip);

  // Our init launch domain will again be isomorphic to our coloring domain.
  Domain launch_domain = full_color_domain;
  ArgumentMap arg_map;

  // First initialize the 'FID_X' and 'FID_Y' fields with some data
  IndexLauncher init_launcher(INIT_FIELD_TASK_ID, launch_domain,
                              TaskArgument(NULL, 0), arg_map);
  init_launcher.add_region_requirement(
      RegionRequirement(full_lp, 0/*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, top_lr));
  init_launcher.add_field(0, FID_X);
  init_launcher.add_field(0, FID_Y);
  init_launcher.add_field(0, FID_VAL);
  runtime->execute_index_space(ctx, init_launcher);

  // Now we launch the computation to calculate the value
  for (int j = 0; j < num_iterations; j++) {
    for (int i = num_waves - 1; i >= 0; i--) {
      DomainPoint compute_point = DomainPoint::from_point<1>(make_point(i));
      DomainPoint data_point = DomainPoint::from_point<1>(make_point(i+1));
      LogicalRegion compute_region =
          runtime->get_logical_subregion_by_color(first_lp, compute_point);
      LogicalRegion data_region =
          runtime->get_logical_subregion_by_color(first_lp, data_point);

      ComputeArgs compute_args;
      compute_args.angle = angle;
      compute_args.parallel_length = parallel_size;
      TaskLauncher helper_launcher(LAUNCHER_HELPER_TASK_ID,
           TaskArgument(&compute_args, sizeof(ComputeArgs)));
      helper_launcher.add_region_requirement(
          RegionRequirement(compute_region,
                            READ_WRITE, EXCLUSIVE, top_lr));
      helper_launcher.add_region_requirement(
          RegionRequirement(data_region,
                            READ_ONLY, EXCLUSIVE, top_lr));
      helper_launcher.add_field(0, FID_VAL);
      helper_launcher.add_field(1, FID_VAL);
      runtime->execute_task(ctx, helper_launcher);
    }

    if (j == 0 && num_iterations > 1) {
      LogicalRegion first_wave_region_intermediate =
        runtime->get_logical_subregion_by_color(first_lp,
            DomainPoint::from_point<1>(make_point(num_waves - 1)));
      LogicalPartition first_wave_intermediate_partition =
        runtime->get_logical_partition_by_color(first_wave_region_intermediate,
            DomainPoint::from_point<1>(make_point(0)));
      LogicalRegion corner_region_first_wave =
        runtime->get_logical_subregion_by_color(first_wave_intermediate_partition,
            DomainPoint::from_point<1>(make_point(0)));

      LogicalRegion last_wave_region_intermediate =
        runtime->get_logical_subregion_by_color(first_lp,
            DomainPoint::from_point<1>(make_point(0)));
      LogicalPartition last_wave_intermediate_partition =
        runtime->get_logical_partition_by_color(last_wave_region_intermediate,
            DomainPoint::from_point<1>(make_point(0)));
      LogicalRegion corner_region_last_wave =
        runtime->get_logical_subregion_by_color(last_wave_intermediate_partition,
            DomainPoint::from_point<1>(make_point(subregions_per_wave - 1)));

      TaskLauncher pause_launcher(PAUSE_TASK_ID,
           TaskArgument(NULL, 0));
      pause_launcher.add_region_requirement(
          RegionRequirement(corner_region_first_wave,
                            READ_WRITE, EXCLUSIVE, top_lr));
      pause_launcher.add_region_requirement(
          RegionRequirement(corner_region_last_wave,
                            READ_WRITE, EXCLUSIVE, top_lr));
      pause_launcher.add_field(0, FID_VAL);
      pause_launcher.add_field(1, FID_VAL);
      runtime->execute_task(ctx, pause_launcher);
    }
  }

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
  runtime->execute_task(ctx, check_launcher);

  // Clean up our region, index space, and field space
  runtime->destroy_logical_region(ctx, top_lr);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_space(ctx, is);
}

// The standard initialize field task from earlier examples
void init_field_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 3);

  std::set<unsigned int>::iterator fields = task->regions[0].privilege_fields.begin();
  FieldID fidx = *fields;
  FieldID fidy = *(++fields);
  FieldID fid_val_write = *(++fields);
  const int pointx = task->index_point.point_data[0];
  const int pointy = task->index_point.point_data[1];
  printf("Initializing fields %d and %d for block (%d, %d)...\n", fidx, fidy, pointx, pointy);

  RegionAccessor<AccessorType::Generic, int> accx = 
    regions[0].get_field_accessor(fidx).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> accy = 
    regions[0].get_field_accessor(fidy).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> acc_val_write = 
    regions[0].get_field_accessor(fid_val_write).typeify<int>();

  Domain dom = runtime->get_index_space_domain(ctx, 
      task->regions[0].region.get_index_space());
  Rect<2> rect = dom.get_rect<2>();
  for (GenericPointInRectIterator<2> pir(rect); pir; pir++)
  {
    accx.write(DomainPoint::from_point<2>(pir.p), pir.p[0]);
    accy.write(DomainPoint::from_point<2>(pir.p), pir.p[1]);
    acc_val_write.write(DomainPoint::from_point<2>(pir.p), 1);
  }
}

void launcher_helper_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->regions[0].privilege_fields.size() == 1);
  assert(task->regions[1].privilege_fields.size() == 1);
  assert(task->arglen == sizeof(ComputeArgs));
  ComputeArgs compute_args = *((ComputeArgs *)task->args);
  //const int angle = compute_args.angle;

  LogicalRegion lr_0 = regions[0].get_logical_region(); // The region to compute
  LogicalRegion lr_1 = regions[1].get_logical_region(); // The data region

  LogicalPartition compute_partition =
      runtime->get_logical_partition_by_color(ctx, lr_0,
                  DomainPoint::from_point<1>(make_point(0)));
  LogicalPartition data_partition = 
      runtime->get_logical_partition_by_color(ctx, lr_1,
                  DomainPoint::from_point<1>(make_point(0)));

  Domain compute_launch_domain =
      runtime->get_index_partition_color_space(compute_partition.get_index_partition());
  
  ArgumentMap arg_map;
  IndexLauncher compute_launcher(COMPUTE_TASK_ID, compute_launch_domain,
       TaskArgument(&compute_args, sizeof(ComputeArgs)), arg_map);
  compute_launcher.add_region_requirement(
      RegionRequirement(data_partition, 0,
                        READ_ONLY, EXCLUSIVE, lr_1));
  compute_launcher.add_region_requirement(
      RegionRequirement(compute_partition, 0,
                        READ_WRITE, EXCLUSIVE, lr_0));
  compute_launcher.add_field(0, FID_VAL);
  compute_launcher.add_field(1, FID_VAL);

  runtime->execute_index_space(ctx, compute_launcher);
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
  assert(task->arglen == sizeof(ComputeArgs));
  ComputeArgs compute_args = *((ComputeArgs *)task->args);
  const int angle = compute_args.angle;
  const int parallel_length  = compute_args.parallel_length;
  
  FieldID val_fid_prev_wave = *(task->regions[0].privilege_fields.begin());
  FieldID val_fid_curr = *(task->regions[1].privilege_fields.begin());

  RegionAccessor<AccessorType::Generic, int> prev_wave_acc = 
    regions[0].get_field_accessor(val_fid_prev_wave).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> curr_acc = 
    regions[1].get_field_accessor(val_fid_curr).typeify<int>();

  Domain dom = runtime->get_index_space_domain(ctx,
      task->regions[1].region.get_index_space());
  Rect<2> rect = dom.get_rect<2>();

  Domain prev_wave_dom = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  size_t prev_wave_volume = prev_wave_dom.get_volume();

  Point<2> lo = rect.lo;
  Point<2> hi = rect.hi;
  Point<2> cur_point;
  int prev_wave_val;
  Point<2> offset_prev_wave;
  int primary_loop_index;
  if (angle == 180) {
    offset_prev_wave = make_point(1,0);
    primary_loop_index = 0;
  }
  else {
    offset_prev_wave = make_point(0,1);
    primary_loop_index = 1;
  }

  for (long int i = hi[primary_loop_index]; i >= lo[primary_loop_index]; i--) {
    for (long int j = hi[1-primary_loop_index]; j >= lo[1-primary_loop_index]; j--) {
      if (angle == 180) {
        cur_point = make_point(i, j);
      }
      else {
        cur_point = make_point(j, i);
      }
      if (i == hi[primary_loop_index]) {
        if (prev_wave_volume > 0) {
          prev_wave_val = prev_wave_acc.read(DomainPoint::from_point<2>(cur_point + offset_prev_wave));
        }
        else {
          // make the starting value be the index of the minor axis (parallel to wave direction)
          prev_wave_val = parallel_length - 1 - j;
        }
      }
      else {
        prev_wave_val = curr_acc.read(DomainPoint::from_point<2>(cur_point + offset_prev_wave));
      }
      int computed_val = prev_wave_val + 1;
      curr_acc.write(DomainPoint::from_point<2>(cur_point), computed_val);
    }
  }
}

void pause_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  unsigned int guess = 0;
  for (int i = 0; i < 10000; i++) {
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

  std::set<unsigned int>::iterator fields = task->regions[0].privilege_fields.begin();
  FieldID fidx = *fields;
  FieldID fidy = *(++fields);
  FieldID fid_val = *(++fields);

  RegionAccessor<AccessorType::Generic, int> accx = 
    regions[0].get_field_accessor(fidx).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> accy = 
    regions[0].get_field_accessor(fidy).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> acc_val = 
    regions[0].get_field_accessor(fid_val).typeify<int>();

  Domain dom = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  Rect<2> rect = dom.get_rect<2>();

  // This is the checking task so we can just do the slow path
  bool all_passed = true;
  for (GenericPointInRectIterator<2> pir(rect); pir; pir++)
  {
    int x = side_length_x - 1 - accx.read(DomainPoint::from_point<2>(pir.p));
    int y = side_length_y - 1 - accy.read(DomainPoint::from_point<2>(pir.p));
    int val = acc_val.read(DomainPoint::from_point<2>(pir.p));
    int expected = x + y + 1;

    //printf("At point (%lld, %lld).  Checking for values %d and %d... expected %d, found %d\n", pir.p[0], pir.p[1], x, y, expected, val);
    
    if (expected != val) {
      all_passed = false;
      break;
    }
  }
  if (all_passed)
    printf("SUCCESS!\n");
  else
    printf("FAILURE!\n");
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  Runtime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/, AUTO_GENERATE_ID, TaskConfigOptions(), "top level task");
  Runtime::register_legion_task<init_field_task>(INIT_FIELD_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/, AUTO_GENERATE_ID, TaskConfigOptions(), "init task");
  Runtime::register_legion_task<launcher_helper_task>(LAUNCHER_HELPER_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/, AUTO_GENERATE_ID, TaskConfigOptions(false,true,false), "launcher helper task");
  Runtime::register_legion_task<compute_task>(COMPUTE_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/, AUTO_GENERATE_ID, TaskConfigOptions(true, false, false), "compute task");
  Runtime::register_legion_task<pause_task>(PAUSE_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/, AUTO_GENERATE_ID, TaskConfigOptions(true, false, false), "pause task");
  Runtime::register_legion_task<check_task>(CHECK_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/, AUTO_GENERATE_ID, TaskConfigOptions(), "check task");

  return Runtime::start(argc, argv);
}
