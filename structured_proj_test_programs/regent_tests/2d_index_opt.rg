import "regent"

local c = terralib.includec("stdio.h")

struct point { x : int, y : int, value : int }

task init_task(points : region(ispace(int2d), point))
where
  writes(points.{x,y,value})
do
  for p in points do
    points[p].x = p.x
    points[p].y = p.y
    points[p].value = 0
  end
end

task calc_task(points : region(ispace(int2d), point)) 
where 
  reads(points.{x,y}), 
  writes(points.value)
do 
  for p in points do
    points[p].value = 0
  end
end

task calc_task_wavefront(points : region(ispace(int2d), point), left_points : region(ispace(int2d), point), bottom_points : region(ispace(int2d), point)) 
where 
  reads(left_points.x, bottom_points.y), 
  writes(points.value)
do 
  for p in points do
    points[p].value = left_points[p].x + bottom_points[p].y
  end
end

task printer(points : region(ispace(int2d), point))
where
  reads(points.{x,y,value})
do
  for p in points do
    c.printf("Point (%d, %d) value is %d.\n",
        points[p].x, points[p].y, points[p].value)
  end
end

task main()
  --c.printf("Running Index tester...\n")

  var chunks = 4

  var points = region(ispace(int2d, {x=4, y=4}, {x=0, y=0}), point)
  var part_ispace = ispace(int2d, {x=4, y=4}, {x=0, y=0})
  var launch_ispace = ispace(int2d, {x=3, y=3}, {x=1, y=1})
  var part = partition(equal, points, part_ispace)

  --for p in points do
    --c.printf("Point (%d, %d) is in points\n", p.x, p.y)
  --end
  --for p in launch_ispace do
    --c.printf("Point (%d, %d) is in the launch index space\n", p.x, p.y)
  --end

  __demand(__parallel)
  for init_p in part_ispace do
    init_task(part[init_p])
  end

  calc_task(part[{x=1, y=1}])

  __demand(__parallel)
  for launch_p in launch_ispace do
    calc_task_wavefront(part[launch_p], part[launch_p - {1, 0}], part[launch_p - {x=0, y=1}])
  end

  --printer(points)

  --c.printf("Done!\n")
end

regentlib.start(main)
