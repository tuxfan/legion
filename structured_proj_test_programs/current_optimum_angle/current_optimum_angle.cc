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
#include "default_mapper.h"
#include <unistd.h>  // for sleep and usleep
using namespace Legion;
using namespace Legion::Mapping;
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
  int num_subregions_x;
  int num_subregions_y;
  int diag_num;
};

class SliceMapper : public DefaultMapper {
public:
  SliceMapper(Machine machine, Runtime *rt, Processor local, int sl);
public:
  virtual void slice_task(const MapperContext ctx,
                          const Task& task,
                          const SliceTaskInput& input,
                                SliceTaskOutput& output);
private:
  void compute_cached_procs(std::vector<Processor> all_procs);
public:
  unsigned slice_side_length;
  bool cached;
  std::vector<Processor> cached_procs;
};

// pass arguments through to Default
SliceMapper::SliceMapper(Machine m, Runtime *rt, Processor p, int sl)
  : DefaultMapper(rt->get_mapper_runtime(), m, p), slice_side_length(sl),
  cached(false)
{
}

void SliceMapper::slice_task(const MapperContext      ctx,
                             const Task&              task,
                             const SliceTaskInput&    input,
                                   SliceTaskOutput&   output)
{
  if (slice_side_length == 0)
  {
    DefaultMapper::slice_task(ctx, task, input, output);
    return;
  }
  // Iterate over all the points and send them all over the world
  //output.slices.resize(input.domain.get_volume());
  unsigned idx = 0;

  Machine::ProcessorQuery all_procs(machine);
  all_procs.only_kind(local_cpus[0].kind());
  std::vector<Processor> procs(all_procs.begin(), all_procs.end());
  //printf("there are %lu all procs size\n", procs.size());
  if (!cached)
  {
    Machine::ProcessorQuery all_procs(machine);
    all_procs.only_kind(local_cpus[0].kind());
    std::vector<Processor> procs(all_procs.begin(), all_procs.end());
    compute_cached_procs(procs);
  }

  switch (input.domain.get_dim())
  {
    case 1:
      {
        // This should only be called for the compute task.
        Rect<1> rect = input.domain;
        Point<1> lo = rect.lo;
        Point<1> hi = rect.hi;
        ComputeArgs compute_args = *((ComputeArgs *)task.args);
        const int diag_num = compute_args.diag_num;
        const int num_subregions_x = compute_args.num_subregions_x;

        int starting_x;
        if (diag_num < num_subregions_x)
        {
          starting_x = num_subregions_x - diag_num - 1 + lo[0];
        }
        else
        {
          starting_x = lo[0];
        }

        int cur_lo = 0;
        int cur_hi =
          ((starting_x/slice_side_length) + 1) * slice_side_length - 1 - starting_x;
        cur_hi = std::min<int>(cur_hi, hi[0]);
        int cur_proc_idx = starting_x / slice_side_length;
        int num_slices = ((hi[0] - lo[0]) / slice_side_length) + 1;
        for (int i = 0; i < num_slices; i++)
        {
          Point<1> slice_lo(cur_lo);
          Point<1> slice_hi(cur_hi);
          Rect<1> slice(slice_lo, slice_hi);
          output.slices.push_back(TaskSlice(slice,
              cached_procs[cur_proc_idx % cached_procs.size()],
              false/*recurse*/, true/*stealable*/));
          cur_lo = cur_hi + 1;
          cur_hi = std::min<int>(cur_hi + slice_side_length, hi[0]);
          cur_proc_idx++;
        }
        break;
      }
    case 2: // this should keep regions in sync between init and compute tasks
      {
        Rect<2> rect = input.domain;
        Point<2> lo = rect.lo;
        Point<2> hi = rect.hi;
        assert((hi[0] - lo[0] + 1) % slice_side_length == 0);
        assert((hi[1] - lo[1] + 1) % slice_side_length == 0);
        unsigned x_iter = (hi[0] - lo[0]) / slice_side_length + 1;
        unsigned y_iter = (hi[1] - lo[1]) / slice_side_length + 1;
        output.slices.resize(x_iter * y_iter);
        //printf("using the slice mapper, side length %u\n", slice_side_length);
        for (unsigned x = 0; x < x_iter; ++x)
        {
          for (unsigned y = 0; y < y_iter; ++y)
          {
            Point<2> slice_lo(x * slice_side_length, y * slice_side_length);
            Point<2> slice_hi((x+1) * slice_side_length - 1, (y+1) * slice_side_length - 1);
            //printf("x = %u, y = %u, slice goes from (%lld, %lld) to (%lld, %lld), mapped to %lu\n",
              //x, y, slice_lo[0], slice_lo[1], slice_hi[0], slice_hi[1], x % cached_procs.size());
            Rect<2> slice(slice_lo, slice_hi);
            output.slices[idx] = TaskSlice(slice, cached_procs[x % cached_procs.size()],
              false/*recurse*/, true/*stealable*/);
            idx++;
          }
        }
        break;
      }
    default:
      assert(false);
  }
}

void SliceMapper::compute_cached_procs(std::vector<Processor> all_procs)
{
  std::vector<AddressSpace> address_spaces;
  for (unsigned i = 0; i < all_procs.size(); ++i)
  {
    Processor p = all_procs[i];
    if (std::find(address_spaces.begin(), address_spaces.end(), p.address_space()) == address_spaces.end())
    {
      address_spaces.push_back(p.address_space());
      cached_procs.push_back(p);
    }
  }
  cached = true;
}

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
      const int diag_num = compute_args.diag_num;
      const int num_subregions_x = compute_args.num_subregions_x;
      const int num_subregions_y = compute_args.num_subregions_y;
      
      Point<2> proj_point;

      if (diag_num < num_subregions_x)
      {
        proj_point[0] = num_subregions_x - diag_num - 1 + point_in_launch;
      }
      else
      {
        proj_point[0] = point_in_launch;
      }
      if (diag_num < num_subregions_x)
      {
        proj_point[1] = num_subregions_y - 1 - point_in_launch;
      }
      else
      {
        proj_point[1] = num_subregions_y - 1 - (diag_num - num_subregions_x + 1) - point_in_launch;
      }

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
      const int diag_num = compute_args.diag_num;
      const int num_subregions_x = compute_args.num_subregions_x;
      const int num_subregions_y = compute_args.num_subregions_y;
      
      Point<2> proj_point;

      if (diag_num < num_subregions_x)
      {
        proj_point[0] = num_subregions_x - diag_num - 1 + point_in_launch;
      }
      else
      {
        proj_point[0] = point_in_launch;
      }
      if (diag_num < num_subregions_x)
      {
        proj_point[1] = num_subregions_y - 1 - point_in_launch;
      }
      else
      {
        proj_point[1] = num_subregions_y - 1 - (diag_num - num_subregions_x + 1) - point_in_launch;
      }

      proj_point[0] += 1;

      LogicalRegion ret_region =
          runtime->get_logical_subregion_by_color(ctx, upper_bound, DomainPoint(proj_point));
      return ret_region;
    }

    virtual unsigned get_depth() const {
      return 0;
    }
};

class YDiffProjectionFunctor : public ProjectionFunctor
{
  public:
    YDiffProjectionFunctor()
      : ProjectionFunctor() {}

    ~YDiffProjectionFunctor() {}

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
      const int diag_num = compute_args.diag_num;
      const int num_subregions_x = compute_args.num_subregions_x;
      const int num_subregions_y = compute_args.num_subregions_y;
      
      Point<2> proj_point;

      if (diag_num < num_subregions_x)
      {
        proj_point[0] = num_subregions_x - diag_num - 1 + point_in_launch;
      }
      else
      {
        proj_point[0] = point_in_launch;
      }
      if (diag_num < num_subregions_x)
      {
        proj_point[1] = num_subregions_y - 1 - point_in_launch;
      }
      else
      {
        proj_point[1] = num_subregions_y - 1 - (diag_num - num_subregions_x + 1) - point_in_launch;
      }

      proj_point[1] += 1;

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

  printf("Running computation for (%d, %d) dimensions...\n",
      side_length_x, side_length_y);
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

  // Now we launch the computation to calculate Pascal's triangle
  int num_diags = num_subregions_x + num_subregions_y - 1;
  int min_dim_subregions;
  if (num_subregions_x < num_subregions_y)
  {
    min_dim_subregions = num_subregions_x;
  }
  else
  {
    min_dim_subregions = num_subregions_y;
  }

  fm.wait_all_results();
  double ts_start = Realm::Clock::current_time_in_microseconds();
  for (int j = 0; j < num_iterations; j++)
  {
    for (int diag = 0; diag < num_diags; diag++)
    {
      ComputeArgs compute_args;
      compute_args.sleep_ms = sleep_ms;
      compute_args.num_subregions_x = num_subregions_x;
      compute_args.num_subregions_y = num_subregions_y;
      compute_args.diag_num = diag;

      int tasks_per_diag = min_dim_subregions;
      if (diag < min_dim_subregions)
      {
        tasks_per_diag = diag + 1;
      }
      if (num_diags - diag < min_dim_subregions)
      {
        tasks_per_diag = num_diags - diag;
      }

      Rect<1> diag_launch_bounds(Point<1>(0),
          Point<1>(tasks_per_diag - 1));
      Domain diag_launch_domain = Domain(diag_launch_bounds);

      IndexLauncher compute_launcher(COMPUTE_TASK_ID, diag_launch_domain,
           TaskArgument(&compute_args, sizeof(ComputeArgs)), arg_map);
      compute_launcher.add_region_requirement(
          RegionRequirement(grid_lp, X_PROJ,
                            READ_ONLY, EXCLUSIVE, top_lr));
      compute_launcher.add_region_requirement(
          RegionRequirement(grid_lp, Y_PROJ,
                            READ_ONLY, EXCLUSIVE, top_lr));
      compute_launcher.add_region_requirement(
          RegionRequirement(grid_lp, ID_PROJ,
                            READ_WRITE, EXCLUSIVE, top_lr));
      compute_launcher.add_field(0, FID_VAL);
      compute_launcher.add_field(1, FID_VAL);
      compute_launcher.add_field(2, FID_VAL);

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

// Compute the value triangle value for each point in the rectangle
void compute_task(const Task *task,
                  const std::vector<PhysicalRegion> &regions,
                  Context ctx, Runtime *runtime)
{
  //const int pointx = task->index_point.point_data[0];
  //const int pointy = task->index_point.point_data[1];
  //printf("Starting the compute task at point (%d, %d) in wave %d at time %lld\n", pointx, pointy, pointx + pointy, Realm::Clock::current_time_in_microseconds());
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
  
  FieldID val_fid_x_diff = *(task->regions[0].privilege_fields.begin());
  FieldID val_fid_y_diff = *(task->regions[1].privilege_fields.begin());
  FieldID val_fid_curr = *(task->regions[2].privilege_fields.begin());

  // Sleep the specified amount
  ComputeArgs compute_args = *((ComputeArgs *)task->args);
  //printf(" I am sleeping for base %d ms\n", compute_args.sleep_ms);
  //printf(" I am sleeping for base %d ms, multiplier %f, for a final value of %d\n", compute_args.sleep_ms, compute_args.sleep_multiplier, compute_args.sleep_ms * compute_args.sleep_multiplier);
  usleep(compute_args.sleep_ms);
  //usleep(compute_args.sleep_ms * 1000);
  //usleep(6250);

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
    int expected = 1;
    expected = x + y + 1;

    //printf("At point (%lld, %lld)\n", (*pir)[0], (*pir)[1]);
    //printf("Checking for values %d and %d... expected %d, found %d\n",
        //x, y, expected, val);
    
    if (expected != val)
    {
      all_passed = false;
    //  break;
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
  int slice_side_length = 1;
  const InputArgs &command_args = Runtime::get_input_args();
  for (int i = 1; i < command_args.argc; i++)
  {
      if (!strcmp(command_args.argv[i],"-ssl"))
      {
        slice_side_length = atoi(command_args.argv[++i]);
      }
  }
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    rt->replace_default_mapper(
        new SliceMapper(machine, rt, *it, slice_side_length), *it);
  }
}

int main(int argc, char **argv)
{
  Processor::Kind top_level_proc = Processor::LOC_PROC;
  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i],"-sm"))
      Runtime::add_registration_callback(mapper_registration);
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
  Runtime::preregister_projection_functor(Y_PROJ, new YDiffProjectionFunctor());
  Runtime::preregister_projection_functor(ID_PROJ, new IDProjectionFunctor());

  return Runtime::start(argc, argv);
}
