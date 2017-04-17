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
  COMPUTE_TASK_ID,
  CHECK_TASK_ID,
};

enum FieldIDs {
  FID_X,
  FID_Y,
  FID_PASCAL_RO,
  FID_PASCAL_VAL,
};

enum ProjIDs {
  X_PROJ = 1,
  Y_PROJ = 2,
  ID_PROJ = 3,
};

class IDProjectionFunctor : public StructuredProjectionFunctor
//class IDProjectionFunctor : public ProjectionFunctor
{
  public:
    IDProjectionFunctor(HighLevelRuntime *rt)
      : StructuredProjectionFunctor(rt) {}
    //IDProjectionFunctor(HighLevelRuntime *rt)
      //: ProjectionFunctor(rt) {}

    ~IDProjectionFunctor() {}

    virtual LogicalRegion project(Context ctx, Task *task, unsigned index,
                                  LogicalRegion upper_bound, const DomainPoint &point)
    {
      assert(0);
    }

    virtual LogicalRegion project(Context ctx, Task *task, unsigned index,
                                  LogicalPartition upper_bound, const DomainPoint &point)
    {
      LogicalRegion ret_region = runtime->get_logical_subregion_by_color(ctx, upper_bound, point);
      return ret_region;
    }

    virtual StructuredProjection project_structured(Context ctx, Task *task)
    {
      StructuredProjectionStep first_step =
        StructuredProjectionStep::make_step(1, 0 , 0, 1, 1, 0);
      StructuredProjection sproj(2, 1);
      sproj.steps.push_back(first_step);
      return sproj;
    }

    virtual unsigned get_depth() const {
      return 0;
    }
};

//class XDiffProjectionFunctor : public ProjectionFunctor
class XDiffProjectionFunctor : public StructuredProjectionFunctor
{
  public:
    XDiffProjectionFunctor(HighLevelRuntime *rt)
      : StructuredProjectionFunctor(rt) {}
    //XDiffProjectionFunctor(HighLevelRuntime *rt)
      //: ProjectionFunctor(rt) {}

    ~XDiffProjectionFunctor() {}

    virtual LogicalRegion project(Context ctx, Task *task, unsigned index,
                                  LogicalRegion upper_bound, const DomainPoint &point)
    {
      assert(0);
    }

    virtual LogicalRegion project(Context ctx, Task *task, unsigned index,
                                  LogicalPartition upper_bound, const DomainPoint &point)
    {
      const Point<2> onex = make_point(1,0);
      const Point<2> new_point = point.get_point<2>() + onex;
      LogicalRegion ret_region = runtime->get_logical_subregion_by_color(ctx, upper_bound, DomainPoint::from_point<2>(new_point));
      return ret_region;
    }

    virtual StructuredProjection project_structured(Context ctx, Task *task)
    {
      StructuredProjectionStep first_step =
        StructuredProjectionStep::make_step(1, 0 , 1, 1, 1, 0);
      StructuredProjection sproj(2, 1);
      sproj.steps.push_back(first_step);
      return sproj;
    }

    virtual unsigned get_depth() const {
      return 0;
    }
};

//class YDiffProjectionFunctor : public ProjectionFunctor
class YDiffProjectionFunctor : public StructuredProjectionFunctor
{
  public:
    YDiffProjectionFunctor(HighLevelRuntime *rt)
      : StructuredProjectionFunctor(rt) {}
    //YDiffProjectionFunctor(HighLevelRuntime *rt)
      //: ProjectionFunctor(rt) {}

    ~YDiffProjectionFunctor() {}

    virtual LogicalRegion project(Context ctx, Task *task, unsigned index,
                                  LogicalRegion upper_bound, const DomainPoint &point)
    {
      assert(0);
    }

    virtual LogicalRegion project(Context ctx, Task *task, unsigned index,
                                  LogicalPartition upper_bound, const DomainPoint &point)
    {
      const Point<2> oney = make_point(0,1);
      const Point<2> new_point = point.get_point<2>() + oney;
      LogicalRegion ret_region = runtime->get_logical_subregion_by_color(ctx, upper_bound, DomainPoint::from_point<2>(new_point));
      return ret_region;
    }

    virtual StructuredProjection project_structured(Context ctx, Task *task)
    {
      StructuredProjectionStep first_step =
        StructuredProjectionStep::make_step(1, 0 , 0, 1, 1, 1);
      StructuredProjection sproj(2, 1);
      sproj.steps.push_back(first_step);
      return sproj;
    }

    virtual unsigned get_depth() const {
      return 0;
    }
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int side_length = 4;
  int num_iterations = 1;
  int num_subregions_side = 4; // Assumed to divide side_length
  // Check for any command line arguments
  {
      const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-n"))
        side_length = 1 << atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-b"))
        num_subregions_side = 1 << atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-i"))
        num_iterations = 1 << atoi(command_args.argv[++i]);
    }
  }

  if (side_length % num_subregions_side != 0) {
    printf("subregions per side must evenly divide side length!\n");
    assert(0);
  }

  printf("Running pascal triangle computation for %d side length...\n", side_length);
  printf("Partitioning data into %d sub-regions per side...\n", num_subregions_side);

  // For this example we'll create a single logical region with two
  // fields.  We'll initialize the field identified by 'FID_X' and 'FID_Y' with
  // our input data and then compute the pascal value and write into 'FID_PASCAL_VAL'.
  Rect<2> elem_rect(make_point(0,0),make_point(side_length-1, side_length-1));
  IndexSpace is = runtime->create_index_space(ctx, 
                          Domain::from_rect<2>(elem_rect));
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(int),FID_X);
    allocator.allocate_field(sizeof(int),FID_Y);
    allocator.allocate_field(sizeof(int),FID_PASCAL_RO);
    allocator.allocate_field(sizeof(int),FID_PASCAL_VAL);
  }
  LogicalRegion pascal_lr = runtime->create_logical_region(ctx, is, fs);
  
  // Make our color_domain based on the number of subregions
  // that we want to create.  We creatre extra empty subregions around the outside of the grid
  // so that the compute tasks can all work the same.
  Rect<2> color_bounds(make_point(0,0),make_point(num_subregions_side, num_subregions_side));
  Domain color_domain = Domain::from_rect<2>(color_bounds);

  // Create (possibly coarser) grid partition of the grid of points
  IndexPartition grid_ip;
  {
    const int points_per_partition = side_length/num_subregions_side;
    DomainPointColoring d_coloring;
    for (Domain::DomainPointIterator itr(color_domain); itr; itr++) {
      // Make the empty bounding subregions.
      if (itr.p[0] == num_subregions_side || itr.p[1] == num_subregions_side) {
          Rect<2> subrect(make_point(0,0),make_point(-1,-1));
          d_coloring[itr.p] = Domain::from_rect<2>(subrect);
          continue;
      }
      int x_start = itr.p[0] * points_per_partition;
      int y_start = itr.p[1] * points_per_partition;
      int x_end = (itr.p[0] + 1) * points_per_partition - 1;
      int y_end = (itr.p[1] + 1) * points_per_partition - 1;
      Rect<2> subrect(make_point(x_start, y_start),make_point(x_end, y_end));
      d_coloring[itr.p] = Domain::from_rect<2>(subrect);
    }


    // Once we've computed our coloring then we can
    // create our partition.
    grid_ip = runtime->create_index_partition(ctx, is, color_domain,
                                    d_coloring);
  }

  // Once we've created our index partitions, we can get the
  // corresponding logical partitions for the pascal_lr
  // logical region.
  LogicalPartition grid_lp = 
    runtime->get_logical_partition(ctx, pascal_lr, grid_ip);

  // Our launch domain will again be only include the data subregions and not the dummy ones
  Rect<2> launch_bounds(make_point(0,0),make_point(num_subregions_side-1,num_subregions_side-1));
  Domain launch_domain = Domain::from_rect<2>(launch_bounds);
  ArgumentMap arg_map;

  // First initialize the 'FID_X' and 'FID_Y' fields with some data
  IndexLauncher init_launcher(INIT_FIELD_TASK_ID, launch_domain,
                              TaskArgument(NULL, 0), arg_map);
  init_launcher.add_region_requirement(
      RegionRequirement(grid_lp, 0/*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, pascal_lr));
  init_launcher.add_field(0, FID_X);
  init_launcher.add_field(0, FID_Y);
  init_launcher.add_field(0, FID_PASCAL_RO);
  init_launcher.add_field(0, FID_PASCAL_VAL);
  runtime->execute_index_space(ctx, init_launcher);

  // Now we launch the computation to calculate Pascal's triangle
  for (int j = 0; j < num_iterations; j++) {
    IndexLauncher compute_launcher(COMPUTE_TASK_ID, launch_domain,
         TaskArgument(NULL, 0), arg_map);
    compute_launcher.add_region_requirement(
        RegionRequirement(grid_lp, X_PROJ,
                          READ_ONLY, EXCLUSIVE, pascal_lr));
    compute_launcher.add_region_requirement(
        RegionRequirement(grid_lp, Y_PROJ,
                          READ_ONLY, EXCLUSIVE, pascal_lr));
    compute_launcher.add_region_requirement(
        RegionRequirement(grid_lp, ID_PROJ,
                          READ_WRITE, EXCLUSIVE, pascal_lr));
    compute_launcher.add_field(0, FID_PASCAL_VAL);
    compute_launcher.add_field(1, FID_PASCAL_VAL);
    compute_launcher.add_field(2, FID_PASCAL_VAL);

    runtime->execute_index_space(ctx, compute_launcher);
  }

  // Finally, we launch a single task to check the results.
  TaskLauncher check_launcher(CHECK_TASK_ID, 
      TaskArgument(&side_length, sizeof(side_length)));
  check_launcher.add_region_requirement(
      RegionRequirement(pascal_lr, READ_ONLY, EXCLUSIVE, pascal_lr));
  check_launcher.add_field(0, FID_X);
  check_launcher.add_field(0, FID_Y);
  check_launcher.add_field(0, FID_PASCAL_VAL);
  runtime->execute_task(ctx, check_launcher);

  // Clean up our region, index space, and field space
  runtime->destroy_logical_region(ctx, pascal_lr);
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
  assert(task->regions[0].privilege_fields.size() == 4);

  std::set<unsigned int>::iterator fields = task->regions[0].privilege_fields.begin();
  FieldID fidx = *fields;
  FieldID fidy = *(++fields);
  FieldID fid_pascal_ro = *(++fields);
  FieldID fid_pascal_write = *(++fields);
  const int pointx = task->index_point.point_data[0];
  const int pointy = task->index_point.point_data[1];
  fprintf(stderr, "Initializing fields %d and %d for block (%d, %d) with region id %d...\n",
      fidx, fidy, pointx, pointy, task->regions[0].region.get_index_space().get_id());

  RegionAccessor<AccessorType::Generic, int> accx = 
    regions[0].get_field_accessor(fidx).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> accy = 
    regions[0].get_field_accessor(fidy).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> acc_pascal_ro = 
    regions[0].get_field_accessor(fid_pascal_ro).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> acc_pascal_write = 
    regions[0].get_field_accessor(fid_pascal_write).typeify<int>();

  Domain dom = runtime->get_index_space_domain(ctx, 
      task->regions[0].region.get_index_space());
  Rect<2> rect = dom.get_rect<2>();
  for (GenericPointInRectIterator<2> pir(rect); pir; pir++)
  {
    fprintf(stderr, "Actually initializing the fields\n");
    accx.write(DomainPoint::from_point<2>(pir.p), pir.p[0]);
    accy.write(DomainPoint::from_point<2>(pir.p), pir.p[1]);
    acc_pascal_write.write(DomainPoint::from_point<2>(pir.p), 1);
    //acc_pascal_write.write(DomainPoint::from_point<2>(pir.p), pir.p[0]*5+pir.p[1]*3);
    //fprintf(stderr, "writing value %d.\n", pir.p[0]*5+pir.p[1]*3);
    if (pir.p[0] == 3 || pir.p[1] == 3) {
      acc_pascal_ro.write(DomainPoint::from_point<2>(pir.p), 1);
    } else {
      int val = 1;
      int small, big;
      int x = 3 - pir.p[0]; // reverse the order so that the natural order of the tasks does not match the expected order
      int y = 3 - pir.p[1];

      if (x >= y) {
        big = x;
        small = y;
      } else {
        big = y;
        small = x;
      }

      for (int i = big + small; i > big; i--) {
        val = val * i;
      }
      for (int i = small; i > 1; i--) {
        val = val / i;
      }
      acc_pascal_ro.write(DomainPoint::from_point<2>(pir.p), val);
    }
  }
}

// Compute the pascal's triangle value for each point in the rectangle
void compute_task(const Task *task,
                  const std::vector<PhysicalRegion> &regions,
                  Context ctx, Runtime *runtime)
{
  /* UNCOMMENT BELOW FOR DEBUG PRINT STATEMENTS

  fprintf(stderr, "Starting the compute task.\n");
  const int pointx = task->index_point.point_data[0];
  const int pointy = task->index_point.point_data[1];
  fprintf(stderr, "At point (%d, %d).  My region is %d.  X Region is %d.  Y Region is %d.\n",
     pointx, pointy,
     task->regions[2].region.get_index_space().get_id(),
     task->regions[0].region.get_index_space().get_id(),
     task->regions[1].region.get_index_space().get_id());*/
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  assert(task->regions[0].privilege_fields.size() == 1);
  assert(task->regions[1].privilege_fields.size() == 1);
  assert(task->regions[2].privilege_fields.size() == 1);
  
  FieldID pascal_fid_x_diff = *(task->regions[0].privilege_fields.begin());
  FieldID pascal_fid_y_diff = *(task->regions[1].privilege_fields.begin());
  FieldID pascal_fid_curr = *(task->regions[2].privilege_fields.begin());

  RegionAccessor<AccessorType::Generic, int> x_diff_acc = 
    regions[0].get_field_accessor(pascal_fid_x_diff).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> y_diff_acc = 
    regions[1].get_field_accessor(pascal_fid_y_diff).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> curr_acc = 
    regions[2].get_field_accessor(pascal_fid_curr).typeify<int>();

  Domain dom = runtime->get_index_space_domain(ctx,
      task->regions[2].region.get_index_space());
  Rect<2> rect = dom.get_rect<2>();

  Domain x_dom = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  Domain y_dom = runtime->get_index_space_domain(ctx,
      task->regions[1].region.get_index_space());
  size_t x_volume = x_dom.get_volume();
  size_t y_volume = y_dom.get_volume();

  Point<2> lo = rect.lo;
  Point<2> hi = rect.hi;
  Point<2> cur_point;
  int x_diff_val, y_diff_val;
  const Point<2> onex = make_point(1,0);
  const Point<2> oney = make_point(0,1);

  for (long int x = hi[0]; x >= lo[0]; x--) {
    for (long int y = hi[1]; y >= lo[1]; y--) {
      cur_point = make_point(x, y);
      if (x == hi[0]) {
        if (x_volume > 0) {
          x_diff_val = x_diff_acc.read(DomainPoint::from_point<2>(cur_point + onex));
        }
        else {
          x_diff_val = 0;
        }
      }
      else {
        x_diff_val = curr_acc.read(DomainPoint::from_point<2>(cur_point + onex));
      }
      if (y == hi[1]) {
        if (y_volume > 0) {
          y_diff_val = y_diff_acc.read(DomainPoint::from_point<2>(cur_point + oney));
        }
        else {
          y_diff_val = 0;
        }
      }
      else {
        y_diff_val = curr_acc.read(DomainPoint::from_point<2>(cur_point + oney));
      }
      int computed_val = 0;
      if (x_diff_val > y_diff_val) {
        computed_val = x_diff_val + 1;
      }
      else {
        computed_val = y_diff_val + 1;
      }
      curr_acc.write(DomainPoint::from_point<2>(cur_point), computed_val);
    }
  }
}

  /*Point<2> hi = rect.hi;
  int x_diff_val, y_diff_val;
  const Point<2> onex = make_point(1,0);
  const Point<2> oney = make_point(0,1);

  for (GenericPointInRectIterator<2> pir(rect); pir; pir++)
  {
    if (pir.p[0] == hi[0]) {
      x_diff_val = x_diff_acc.read(DomainPoint::from_point<2>(pir.p + onex));
    }
    else {
      x_diff_val = curr_acc.read(DomainPoint::from_point<2>(pir.p + onex));
    }
    if (pir.p[0] == hi[0]) {
      y_diff_val = y_diff_acc.read(DomainPoint::from_point<2>(pir.p + oney));
    }
    else {
      y_diff_val = curr_acc.read(DomainPoint::from_point<2>(pir.p + oney));
    }
    fprintf(stderr, "read the value %d for x and %d for y.\n", x_diff_val, y_diff_val);
    curr_acc.write(DomainPoint::from_point<2>(pir.p), x_diff_val + y_diff_val);
    fprintf(stderr, "writing the value %d.\n", x_diff_val + y_diff_val);
  }
  fprintf(stderr, "Finishing the compute task.\n");
}*/

void check_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 3);
  assert(task->arglen == sizeof(int));
  const int side_length = *((const int*)task->args);  

  std::set<unsigned int>::iterator fields = task->regions[0].privilege_fields.begin();
  FieldID fidx = *fields;
  FieldID fidy = *(++fields);
  FieldID fid_pascal = *(++fields);

  RegionAccessor<AccessorType::Generic, int> accx = 
    regions[0].get_field_accessor(fidx).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> accy = 
    regions[0].get_field_accessor(fidy).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> acc_pascal = 
    regions[0].get_field_accessor(fid_pascal).typeify<int>();

  Domain dom = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  Rect<2> rect = dom.get_rect<2>();

  // This is the checking task so we can just do the slow path
  bool all_passed = true;
  for (GenericPointInRectIterator<2> pir(rect); pir; pir++)
  {
    int x = side_length - 1 - accx.read(DomainPoint::from_point<2>(pir.p));
    int y = side_length - 1 - accy.read(DomainPoint::from_point<2>(pir.p));
    int pascal = acc_pascal.read(DomainPoint::from_point<2>(pir.p));
    int expected = 1;
    expected = x + y + 1;

    printf("At point (%lld, %lld)\n", pir.p[0], pir.p[1]);
    printf("Checking for values %d and %d... expected %d, found %d\n", x, y, expected, pascal);
    
    if (expected != pascal) {
      all_passed = false;
      //break;
    }
  }
  if (all_passed)
    printf("SUCCESS!\n");
  else
    printf("FAILURE!\n");
}

void registration_callback(Machine machine, HighLevelRuntime *rt,
                               const std::set<Processor> &local_procs)
{
  rt->register_projection_functor(X_PROJ, new XDiffProjectionFunctor(rt));
  rt->register_projection_functor(Y_PROJ, new YDiffProjectionFunctor(rt));
  rt->register_projection_functor(ID_PROJ, new IDProjectionFunctor(rt));
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  Runtime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/, AUTO_GENERATE_ID, TaskConfigOptions(), "top level task");
  Runtime::register_legion_task<init_field_task>(INIT_FIELD_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/, AUTO_GENERATE_ID, TaskConfigOptions(), "init task");
  Runtime::register_legion_task<compute_task>(COMPUTE_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/, AUTO_GENERATE_ID, TaskConfigOptions(), "compute task");
  Runtime::register_legion_task<check_task>(CHECK_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/, AUTO_GENERATE_ID, TaskConfigOptions(), "check task");
  HighLevelRuntime::set_registration_callback(registration_callback);

  return Runtime::start(argc, argv);
}
