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
#include <random> // for gaussian distribution of sleep

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
  INIT_TASK_ID,
  MULT_TASK_ID,
  REDUCE_TASK_ID,
  CHECK_TASK_ID,
};

enum FieldIDs {
  FID_VAL,
};

enum ReductionIDs {
  SUM_REDUCE_ID = 1,
};

enum OrderingIDs {
  TIME_ORDERING_ID = 1,
};

enum ProjIDs {
  MAT_1_PROJ_ID = 1,
  MAT_2_PROJ_ID = 2,
  MAT_3_PROJ_ID = 3,
};

struct CheckTaskArgs {
  bool print_matrices;
};

struct ReduceTaskArgs {
  int mod_value;
  int subregion_cols;
};

struct ComputeArgs {
  int angle;
  int parallel_length;
  int sleep_ms;
  double sleep_multiplier;
};

// Reduction Op
class SumReduction {
public:
  typedef int LHS;
  typedef int RHS;
  static const int identity = 0;

  template <bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs)
  {
    lhs += rhs;
  }

  template <bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }
};

// A new mapper to control how the index space is sliced
// for testing out performance and correctness with interslice
// dependencies
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
  // Iterate over all the points and send them all over the world
  //output.slices.resize(input.domain.get_volume());
  unsigned idx = 0;

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
        assert(0);
        Rect<1> rect = input.domain;
        for (PointInRectIterator<1> pir(rect);
              pir(); pir++, idx++)
        {
          Rect<1> slice(*pir, *pir);
          output.slices[idx] = TaskSlice(slice,
              cached_procs[idx % cached_procs.size()],
              false/*recurse*/, true/*stealable*/);
        }
        break;
      }
    case 2: // only care about this case for now
      {
        assert(0);
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
    case 3:
      {
        Rect<3> rect = input.domain;
        for (PointInRectIterator<3> pir(rect);
              pir(); pir++, idx++)
        {
          Rect<3> slice_rect(*pir, *pir);
          TaskSlice slice;
          slice.domain = Domain(slice_rect);
          slice.proc = cached_procs[idx % cached_procs.size()];
          slice.recurse = false;
          slice.stealable = true;
          output.slices.push_back(slice);
        }
        break;
      }
    case 4:
      {
        Rect<4> rect = input.domain;
        for (PointInRectIterator<4> pir(rect);
              pir(); pir++, idx++)
        {
          Rect<4> slice_rect(*pir, *pir);
          TaskSlice slice;
          slice.domain = Domain(slice_rect);
          slice.proc = cached_procs[idx % cached_procs.size()];
          slice.recurse = false;
          slice.stealable = true;
          output.slices.push_back(slice);
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

// Order the computation by the iteration of the "time" loop
class TimeOrderingFunctor : public StructuredOrderingFunctor
{
  public:
    TimeOrderingFunctor()
      : StructuredOrderingFunctor(get_coefficients()) {}

    ~TimeOrderingFunctor() {}

  private:
    static DomainPoint get_coefficients(void)
    {
      DomainPoint coeffs;
      coeffs.dim = 4;
      coeffs[0] = 1;
      coeffs[1] = 1;
      coeffs[2] = 1;
      coeffs[3] = 1;
      return coeffs;
    }
};

class LeftMatrixProjectionFunctor : public StructuredProjectionFunctor
{
  public:
    LeftMatrixProjectionFunctor(int k_coeff, int mod_by)
      : StructuredProjectionFunctor(), k_coeff(k_coeff), mod_by(mod_by) {}

    ~LeftMatrixProjectionFunctor() {}

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
      assert(0);
    }

    virtual StructuredProjection project_structured(Context ctx)
    {
      assert(0);
    }

    virtual AffineStructuredProjection affine_project_structured(Context ctx)
    {
      Transform<3,4> transform;
      transform[0][0] = 1;
      transform[0][1] = 0;
      transform[0][2] = 0;
      transform[0][3] = 0;

      transform[1][0] = 1;
      transform[1][1] = 1;
      transform[1][2] = -1*k_coeff;
      transform[1][3] = -1;

      transform[2][0] = 0;
      transform[2][1] = 0;
      transform[2][2] = 1;
      transform[2][3] = 0;

      Point<3> offset(0, 0, 0);
      Point<3> div_point(1, 1, 1);
      Point<3> mod_point(0, mod_by, 0);
      AffineStructuredProjectionStep first_step =
        AffineStructuredProjectionStep(DomainTransform(transform),
                                       DomainPoint(offset));
      first_step.divisor = DomainPoint(div_point);
      first_step.mod_divisor = DomainPoint(mod_point);
      AffineStructuredProjection sproj;
      sproj.steps.push_back(first_step);
      return sproj;
    }

    virtual unsigned get_depth() const {
      return 0;
    }

    private:
      int k_coeff;
      int mod_by;
};

class RightMatrixProjectionFunctor : public StructuredProjectionFunctor
{
  public:
    RightMatrixProjectionFunctor(int k_coeff, int mod_by)
      : StructuredProjectionFunctor(), k_coeff(k_coeff), mod_by(mod_by) {}

    ~RightMatrixProjectionFunctor() {}

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
      assert(0);
    }

    virtual StructuredProjection project_structured(Context ctx)
    {
      assert(0);
    }

    virtual AffineStructuredProjection affine_project_structured(Context ctx)
    {
      Transform<3,4> transform;
      transform[0][0] = 1;
      transform[0][1] = 1;
      transform[0][2] = -1*k_coeff;
      transform[0][3] = -1;

      transform[1][0] = 0;
      transform[1][1] = 1;
      transform[1][2] = 0;
      transform[1][3] = 0;

      transform[2][0] = 0;
      transform[2][1] = 0;
      transform[2][2] = 1;
      transform[2][3] = 0;

      Point<3> offset(0, 0, 0);
      Point<3> div_point(1, 1, 1);
      Point<3> mod_point(mod_by, 0, 0);
      AffineStructuredProjectionStep first_step =
        AffineStructuredProjectionStep(DomainTransform(transform),
                                       DomainPoint(offset));
      first_step.divisor = DomainPoint(div_point);
      first_step.mod_divisor = DomainPoint(mod_point);
      AffineStructuredProjection sproj;
      sproj.steps.push_back(first_step);
      return sproj;
    }

    virtual unsigned get_depth() const {
      return 0;
    }

    private:
      int k_coeff;
      int mod_by;
};

class ProductMatrixProjectionFunctor : public StructuredProjectionFunctor
{
  public:
    ProductMatrixProjectionFunctor(int k_coeff, int mod_by)
      : StructuredProjectionFunctor(), k_coeff(k_coeff), mod_by(mod_by) {}

    ~ProductMatrixProjectionFunctor() {}

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
      assert(0);
    }

    virtual StructuredProjection project_structured(Context ctx)
    {
      assert(0);
    }

    virtual AffineStructuredProjection affine_project_structured(Context ctx)
    {
      Transform<3,4> transform;
      transform[0][0] = 1;
      transform[0][1] = 0;
      transform[0][2] = 0;
      transform[0][3] = 0;

      transform[1][0] = 0;
      transform[1][1] = 1;
      transform[1][2] = 1;
      transform[1][3] = 0;

      transform[2][0] = 0;
      transform[2][1] = 0;
      transform[2][2] = 1;
      transform[2][3] = 0;

      Point<3> offset(0, 0, 0);
      Point<3> div_point(1, 1, 1);
      Point<3> mod_point(0, mod_by, 0);
      AffineStructuredProjectionStep first_step =
        AffineStructuredProjectionStep(DomainTransform(transform),
                                       DomainPoint(offset));
      first_step.divisor = DomainPoint(div_point);
      first_step.mod_divisor = DomainPoint(mod_point);
      AffineStructuredProjection sproj;
      sproj.steps.push_back(first_step);
      return sproj;
    }

    virtual unsigned get_depth() const {
      return 0;
    }

    private:
      int k_coeff;
      int mod_by;
};

// Little helper method to find the square roots from the algorithm.
// Quite inefficient but who cares.
int find_sq_rt(int value)
{
  for (int i = 0; i * i <= value; i++)
  {
    if (i * i ==  value)
    {
      return i;
    }
  }
  printf("Could not find square root for %d\n", value);
  assert (0);
  return 0;
}

inline int real_mod(int dividend, int divisor)
{
  if (dividend >= 0)
    return dividend % divisor;
  else
    return divisor - (-1 * dividend % divisor);
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_rows = 4;
  int num_cols = 4;
  int num_iterations = 1;
  int num_processors = 8; // This is "processors" from the paper
  int c_val = 2; // This is the c value from the paper
  int sleep_ms = 0; // how long each task should sleep in milliseconds
  bool print_matrices = false;

  // Check for any command line arguments
  {
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-n"))
      {
        num_rows = 1 << atoi(command_args.argv[++i]);
        num_cols = num_rows;
      }
      if (!strcmp(command_args.argv[i],"-p"))
        num_processors = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-c"))
        c_val = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-i"))
        num_iterations = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-sms"))
        sleep_ms = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"--print"))
        print_matrices = true;
    }
  }

  if (num_processors % (c_val * c_val * c_val) != 0)
  {
    printf("p = %d, c = %d.  c^3 must evenly divide p\n", num_processors, c_val);
    assert(0);
  }

  int root_p_c = find_sq_rt(num_processors/c_val);
  int root_p_c3 = find_sq_rt(num_processors/(c_val*c_val*c_val));

  if (num_rows % root_p_c != 0 ||
      num_cols % root_p_c != 0)
  {
    printf("root(p/c) must evenly divide side length!\n");
    assert(0);
  }

  int num_subregions_rows = root_p_c;
  int num_subregions_cols = root_p_c;

  int rows_per_subregion = num_rows / num_subregions_rows;
  int cols_per_subregion = num_rows / num_subregions_cols;

  if (num_rows != num_cols)
  {
    printf("currently only works for square matrices\n");
    assert(0);
  }

  printf("Running computation for (%d, %d) dimensions, p = %d, c = %d...\n",
      num_rows, num_cols, num_processors, c_val);
  printf("Partitioning data into (%d, %d) sub-regions...\n",
      num_subregions_rows, num_subregions_cols);

  // We will create a logical region for each matrix
  Rect<3> elem_rect(Point<3>(0,0,0), Point<3>(num_rows-1,
      num_cols-1, c_val-1));
  IndexSpace is = runtime->create_index_space(ctx, elem_rect);
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(int),FID_VAL);
  }
  LogicalRegion m1_lr = runtime->create_logical_region(ctx, is, fs);
  LogicalRegion m2_lr = runtime->create_logical_region(ctx, is, fs);
  LogicalRegion m3_lr = runtime->create_logical_region(ctx, is, fs);

  // Make our color_domain based on the number of subregions
  // that we want to create.
  Rect<3> color_bounds(Point<3>(0,0,0),
      Point<3>(num_subregions_rows-1, num_subregions_cols-1,c_val-1));
  IndexSpace color_is = runtime->create_index_space(ctx, color_bounds);
  Domain color_domain = runtime->get_index_space_domain(ctx, color_is);

  Transform<3,3> transform;
  transform[0][0] = rows_per_subregion;
  transform[0][1] = 0;
  transform[0][2] = 0;
  transform[1][0] = 0;
  transform[1][1] = cols_per_subregion;
  transform[1][2] = 0;
  transform[2][0] = 0;
  transform[2][1] = 0;
  transform[2][2] = 1;

  Rect<3> extent_rect(Point<3>(0,0,0),
      Point<3>(rows_per_subregion - 1, cols_per_subregion - 1, 0));
  Domain extent(extent_rect);

  IndexPartition grid_ip =
      runtime->create_partition_by_restriction(ctx,
                                               is,
                                               color_is,
                                               DomainTransform(transform),
                                               extent,
                                               DISJOINT_KIND);

  // Once we've created our index partitions, we can get the
  // corresponding logical partitions for the top_lr
  // logical region.
  LogicalPartition m1_grid_lp = 
    runtime->get_logical_partition(ctx, m1_lr, grid_ip);
  LogicalPartition m2_grid_lp = 
    runtime->get_logical_partition(ctx, m2_lr, grid_ip);
  LogicalPartition m3_grid_lp = 
    runtime->get_logical_partition(ctx, m3_lr, grid_ip);

  ArgumentMap arg_map;
  IndexLauncher init_launcher(INIT_TASK_ID, color_domain,
                              TaskArgument(NULL, 0), arg_map);
  init_launcher.add_region_requirement(
      RegionRequirement(m1_grid_lp, 0/*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, m1_lr));
  init_launcher.add_region_requirement(
      RegionRequirement(m2_grid_lp, 0/*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, m2_lr));
  init_launcher.add_region_requirement(
      RegionRequirement(m3_grid_lp, 0/*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, m3_lr));
  init_launcher.add_field(0, FID_VAL);
  init_launcher.add_field(1, FID_VAL);
  init_launcher.add_field(2, FID_VAL);
  runtime->execute_index_space(ctx, init_launcher);

  DomainPoint launch_lo, launch_hi;
  launch_lo.dim = 4;
  launch_hi.dim = 4;

  launch_lo[0] = 0;
  launch_lo[1] = 0;
  launch_lo[2] = 0;
  launch_lo[3] = 0;

  launch_hi[0] = num_subregions_rows - 1;
  launch_hi[1] = num_subregions_cols - 1;
  launch_hi[2] = c_val - 1;
  launch_hi[3] = root_p_c3 - 1;

  Domain mult_launch_domain(launch_lo, launch_hi);

  double ts_start = Realm::Clock::current_time_in_microseconds();
  for (int iter = 0; iter < num_iterations; iter++)
  {
    IndexLauncher mult_launcher(MULT_TASK_ID, mult_launch_domain,
                                TaskArgument(NULL, 0), arg_map);
    mult_launcher.set_ordering_id(TIME_ORDERING_ID);
    mult_launcher.add_region_requirement(
        RegionRequirement(m1_grid_lp, MAT_1_PROJ_ID,
                          READ_ONLY, EXCLUSIVE, m1_lr));
    mult_launcher.add_region_requirement(
        RegionRequirement(m2_grid_lp, MAT_2_PROJ_ID,
                          READ_ONLY, EXCLUSIVE, m2_lr));
    mult_launcher.add_region_requirement(
        RegionRequirement(m3_grid_lp, MAT_3_PROJ_ID,
                          READ_WRITE, EXCLUSIVE, m3_lr));
    //mult_launcher.add_region_requirement(
        //RegionRequirement(m3_grid_lp, MAT_3_PROJ_ID,
                          //SUM_REDUCE_ID, EXCLUSIVE, m3_lr));
    mult_launcher.add_field(0, FID_VAL);
    mult_launcher.add_field(1, FID_VAL);
    mult_launcher.add_field(2, FID_VAL);
    runtime->execute_index_space(ctx, mult_launcher);
  }

  // to time accurately, need to wait on the future map here
  double ts_end = Realm::Clock::current_time_in_microseconds();
  double sim_time = 1e-6 * (ts_end - ts_start);
  printf("ELAPSED TIME = %7.7f s\n", sim_time);

  ReduceTaskArgs reduce_task_args;
  reduce_task_args.mod_value = num_cols;
  reduce_task_args.subregion_cols = num_cols / num_subregions_cols;

  TaskLauncher reduce_launcher(REDUCE_TASK_ID,
      TaskArgument(&reduce_task_args, sizeof(ReduceTaskArgs)));
  reduce_launcher.add_region_requirement(
      RegionRequirement(m3_lr, READ_ONLY, EXCLUSIVE, m3_lr));
  reduce_launcher.add_field(0, FID_VAL);
  runtime->execute_task(ctx, reduce_launcher);

  // Finally, we launch a single task to check the results.
  CheckTaskArgs check_task_args;
  check_task_args.print_matrices = print_matrices;

  TaskLauncher check_launcher(CHECK_TASK_ID,
      TaskArgument(&check_task_args, sizeof(CheckTaskArgs)));
  check_launcher.add_region_requirement(
      RegionRequirement(m1_lr, READ_ONLY, EXCLUSIVE, m1_lr));
  check_launcher.add_region_requirement(
      RegionRequirement(m2_lr, READ_ONLY, EXCLUSIVE, m2_lr));
  check_launcher.add_region_requirement(
      RegionRequirement(m3_lr, READ_ONLY, EXCLUSIVE, m3_lr));
  check_launcher.add_field(0, FID_VAL);
  check_launcher.add_field(1, FID_VAL);
  check_launcher.add_field(2, FID_VAL);
  runtime->execute_task(ctx, check_launcher);

  // Clean up our region, index space, and field space
  runtime->destroy_logical_region(ctx, m1_lr);
  runtime->destroy_logical_region(ctx, m2_lr);
  runtime->destroy_logical_region(ctx, m3_lr);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_space(ctx, is);
}

// The standard initialize field task from earlier examples
void init_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  assert(regions.size() == 3); 
  assert(task->regions.size() == 3);
  assert(task->regions[0].privilege_fields.size() == 1);
  assert(task->regions[1].privilege_fields.size() == 1);
  assert(task->regions[2].privilege_fields.size() == 1);

  std::set<unsigned int>::iterator fields =
      task->regions[0].privilege_fields.begin();
  FieldID mat1_vals = *fields;
  fields = task->regions[1].privilege_fields.begin();
  FieldID mat2_vals = *fields;
  fields = task->regions[2].privilege_fields.begin();
  FieldID mat3_vals = *fields;

  RegionAccessor<AccessorType::Generic, int> acc_m1 = 
    regions[0].get_field_accessor(mat1_vals).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> acc_m2 = 
    regions[1].get_field_accessor(mat2_vals).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> acc_m3 = 
    regions[2].get_field_accessor(mat3_vals).typeify<int>();

  Rect<3> rect = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  //printf("calling the init method\n");
  for (PointInRectIterator<3> pir(rect); pir(); pir++)
  {
    //printf("writing at point (%lld, %lld)\n", (*pir)[0], (*pir)[1]);
    acc_m1.write(*pir, (*pir)[0] + (*pir)[1]);
    acc_m2.write(*pir, (*pir)[0] - (*pir)[1]);
    acc_m3.write(*pir, 0);
  }
  //printf("\n\n");
}

// Compute the value for each point in the rectangle, takes 3 regions
// for an angular sweep
void mult_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  assert(task->regions[0].privilege_fields.size() == 1);
  assert(task->regions[1].privilege_fields.size() == 1);
  assert(task->regions[2].privilege_fields.size() == 1);
  
  FieldID m1_field_val = *(task->regions[0].privilege_fields.begin());
  FieldID m2_field_val = *(task->regions[1].privilege_fields.begin());
  FieldID m3_field_val = *(task->regions[2].privilege_fields.begin());

  // Sleep the specified amount
  //ComputeArgs compute_args = *((ComputeArgs *)task->args);
  //printf(" I am sleeping for base %d ms\n", compute_args.sleep_ms);
  //printf(" I am sleeping for base %d ms, multiplier %f, for a final value of %d\n", compute_args.sleep_ms, compute_args.sleep_multiplier, compute_args.sleep_ms * compute_args.sleep_multiplier);
  //usleep(compute_args.sleep_ms * 1000);

  RegionAccessor<AccessorType::Generic, int> acc_m1 = 
    regions[0].get_field_accessor(m1_field_val).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> acc_m2 = 
    regions[1].get_field_accessor(m2_field_val).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> acc_m3 = 
    regions[2].get_field_accessor(m3_field_val).typeify<int>();

  Rect<3> m1_rect = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  Rect<3> m2_rect = runtime->get_index_space_domain(ctx,
      task->regions[1].region.get_index_space());
  Rect<3> m3_rect = runtime->get_index_space_domain(ctx,
      task->regions[2].region.get_index_space());

  Point<3> m1_lo = m1_rect.lo;
  Point<3> m1_hi = m1_rect.hi;
  Point<3> m2_lo = m2_rect.lo;
  Point<3> m3_lo = m3_rect.lo;

  // ONLY FOR SQUARE MATRICES
  long long int dim = m1_hi[1] - m1_lo[1] + 1;

  // Not used in computation, just let's us put data back where
  // Legion expects it.
  long long int c_height = m1_lo[2];

  for (long long int row = 0; row < dim; row++)
  {
    for (long long int col = 0; col < dim; col++)
    {
      int value = 0;
      int m1_row = m1_lo[0] + row;
      int m1_col = m1_lo[1];
      int m2_row = m2_lo[0];
      int m2_col = m2_lo[1] + col;
      for (long long int k = 0; k < dim; k++)
      {
        Point<3> m1_point(m1_row, m1_col + k, c_height);
        Point<3> m2_point(m2_row + k, m2_col, c_height);
        //printf("at (%lld, %lld), (%lld, %lld): read values %d and %d\n",
            //m1_row, m1_col+k, m2_row+k, m2_col,
            //acc_m1.read(m1_point), acc_m2.read(m2_point));
        value += acc_m1.read(m1_point) * acc_m2.read(m2_point);
      }
      //printf("at point (%lld, %lld), writing val %d\n",
          //m1_row, m2_col, value);
      Point<3> m3_point(m1_row, m3_lo[1] + col, c_height);

      // UNCOMMENT THIS FOR READ/WRITE priveleges and add to value below
      //int cur_value = acc_m3.read(m3_point);
      acc_m3.write(m3_point, acc_m3.read(m3_point) + value);
    }
  }
  //printf("\n\n");
}

void reduce_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime)
{
  ReduceTaskArgs task_args = *((ReduceTaskArgs *)task->args);
  int mod_value = task_args.mod_value;
  int subregion_cols = task_args.subregion_cols;

  FieldID vals_field = *(task->regions[0].privilege_fields.begin());

  RegionAccessor<AccessorType::Generic, int> mat_acc = 
    regions[0].get_field_accessor(vals_field).typeify<int>();

  Rect<3> rect = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());

  Point<3> lo = rect.lo;
  Point<3> hi = rect.hi;

  for (long long int row = lo[0]; row < hi[0]+1; row++)
  {
    for (long long int col = lo[1]; col < hi[1]+1; col++)
    {
      int sum = 0;
      for (long long int height = lo[2]; height < hi[2]+1; height++)
      {
        for (unsigned r = 0; r < regions.size(); r++)
        {
          Point<3> mat_point(row,
              (col + height * subregion_cols) % mod_value, height);
          sum += mat_acc.read(mat_point);
        }
      }
      mat_acc.write(Point<3>(row, col, 0), sum);
    }
  }
}

void check_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  assert(task->regions[0].privilege_fields.size() == 1);
  assert(task->regions[1].privilege_fields.size() == 1);
  assert(task->regions[2].privilege_fields.size() == 1);

  CheckTaskArgs task_args = *((CheckTaskArgs *)task->args);
  bool print_matrices = task_args.print_matrices;

  std::set<unsigned int>::iterator fields =
      task->regions[0].privilege_fields.begin();
  FieldID mat1_vals = *fields;
  fields = task->regions[1].privilege_fields.begin();
  FieldID mat2_vals = *fields;
  fields = task->regions[2].privilege_fields.begin();
  FieldID mat3_vals = *fields;

  RegionAccessor<AccessorType::Generic, int> acc_m1 = 
    regions[0].get_field_accessor(mat1_vals).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> acc_m2 = 
    regions[1].get_field_accessor(mat2_vals).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> acc_m3 = 
    regions[2].get_field_accessor(mat3_vals).typeify<int>();

  Rect<3> rect = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());

  Point<3> lo = rect.lo;
  Point<3> hi = rect.hi;

  bool is_correct = true;
  
  // assuming square for now
  long long int inner_dim = hi[1] - lo[1] + 1;

  for (long long int row = lo[0]; row < hi[0]+1; row++)
  {
    for (long long int col = lo[1]; col < hi[1]+1; col++)
    {
      Point<3> m3_point(row, col, 0);
      int value = 0;
      for (long long int k = 0; k < inner_dim; k++)
      {
        Point<3> m1_point(row, k, 0);
        Point<3> m2_point(k, col, 0);
        value += acc_m1.read(m1_point) * acc_m2.read(m2_point);
      }
      int m3_value = acc_m3.read(m3_point);
      if (m3_value != value)
      {
        printf("Incorrect value at index (%lld, %lld).  Found %d, expected %d\n",
            row, col, m3_value, value);
        is_correct = false;
        assert(print_matrices);
      }
    }
  }

  if (is_correct)
  {
    printf("SUCCESS!!!\n\n\n");
  }
  else
  {
    printf("FAILURE!!!\n\n\n");
  }

  if (print_matrices)
  {
    printf("M1:\n");
    for (long long int row = lo[0]; row < hi[0]+1; row++)
    {
      for (long long int col = lo[1]; col < hi[1]+1; col++)
      {
        int value = acc_m1.read(Point<3>(row, col, 0));
        printf("%d ", value);
      }
      printf("\n");
    }
    printf("\n");
    printf("\n");
    printf("M2:\n");
    for (long long int row = lo[0]; row < hi[0]+1; row++)
    {
      for (long long int col = lo[1]; col < hi[1]+1; col++)
      {
        int value = acc_m2.read(Point<3>(row, col, 0));
        printf("%d ", value);
      }
      printf("\n");
    }
    printf("\n");
    printf("\n");
    printf("M3:\n");
    for (long long int height = lo[2]; height < hi[2]+1; height++)
    {
      for (long long int row = lo[0]; row < hi[0]+1; row++)
      {
        for (long long int col = lo[1]; col < hi[1]+1; col++)
        {
          int value = acc_m3.read(Point<3>(row, col, height));
          printf("%d ", value);
        }
        printf("\n");
      }
      printf("\n");
    }
  }
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

void registration_callback(Machine machine, HighLevelRuntime *rt,
                               const std::set<Processor> &local_procs)
{
  rt->register_ordering_functor(
      TIME_ORDERING_ID , new TimeOrderingFunctor());
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  Runtime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/, AUTO_GENERATE_ID,
      TaskConfigOptions(), "top level task");
  Runtime::register_legion_task<init_task>(INIT_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/, AUTO_GENERATE_ID,
      TaskConfigOptions(true,false,false), "init task");
  Runtime::register_legion_task<mult_task>(MULT_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/, AUTO_GENERATE_ID,
      TaskConfigOptions(true,false,false), "matrix multiply task");
  Runtime::register_legion_task<reduce_task>(REDUCE_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/, AUTO_GENERATE_ID,
      TaskConfigOptions(true,false,false), "manual reduction task");
  Runtime::register_legion_task<check_task>(CHECK_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/, AUTO_GENERATE_ID,
      TaskConfigOptions(true,false,false), "check task");

  // If using the slicing mapper, add it's callback
  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i],"-sm"))
      Runtime::add_registration_callback(mapper_registration);
  }

  Runtime::register_reduction_op<SumReduction>(SUM_REDUCE_ID);

  // This is SUPER ugly, but the fastest thing I can think to do right
  // now if just recompute these two values here (including parsing args again).
  {
    int num_processors = 8;
    int c_val = 2;
    for (int i = 1; i < argc; i++)
    {
      if (!strcmp(argv[i],"-p"))
        num_processors = atoi(argv[++i]);
      if (!strcmp(argv[i],"-c"))
        c_val = atoi(argv[++i]);
    }

    int root_p_c = find_sq_rt(num_processors/c_val);
    int root_p_c3 = find_sq_rt(num_processors/(c_val*c_val*c_val));

    Runtime::preregister_projection_functor(MAT_1_PROJ_ID,
        new LeftMatrixProjectionFunctor(root_p_c3, root_p_c));
    Runtime::preregister_projection_functor(MAT_2_PROJ_ID,
        new RightMatrixProjectionFunctor(root_p_c3, root_p_c));
    Runtime::preregister_projection_functor(MAT_3_PROJ_ID,
        new ProductMatrixProjectionFunctor(root_p_c3, root_p_c));
  }

  // Add the callback for the projection function
  HighLevelRuntime::set_registration_callback(registration_callback);

  return Runtime::start(argc, argv);
}


/*
TIMING CODE:

  double ts_start = Realm::Clock::current_time_in_microseconds();
  for (int i = 0; i < MAX_ITERATION; i++)
  {
    iteration = i;
    BFSTask bfs_task(graph.in_pointer_lp, graph.in_pointer,
                     graph.in_vtx_lp, graph.in_vtx,
                     graph.in_index_lp, graph.in_index,
                     graph.dist_lp[(i+1)%2], graph.dist[(i+1)%2],
                     graph.dist[i%2], task_space, local_args, &iteration);
    fm = runtime->execute_index_space(ctx, bfs_task);
  }
  fm.wait_all_results();
  double ts_end = Realm::Clock::current_time_in_microseconds();
  double sim_time = 1e-6 * (ts_end - ts_start);
  printf("ELAPSED TIME = %7.7f s\n", sim_time);
*/
