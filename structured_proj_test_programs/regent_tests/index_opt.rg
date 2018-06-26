import "regent"

local c = terralib.includec("stdio.h")

struct point { x : int, y : int, value : int }

task init_task(points : region(point))
where
  writes(points.{x,y})
do
  for p in points do
    points[p].x = p
    points[p].y = p
  end
end

task calc_task(points : region(point)) 
where 
  reads(points.{x,y}), 
  writes(points.value)
do 
  for p in points do
    points[p].value = points[p].x + points[p].y
  end
end

task calc_task_left(points : region(point), left_points : region(point)) 
where 
  reads(points.x, left_points.y), 
  writes(points.value)
do 
  for p in points do
    points[p].value = points[p].x + left_points[p].y
  end
end

task printer(points : region(point))
where
  reads(points.{x,y,value})
do
  for p in points do
    c.printf("Point (%d, %d) value is %d.\n",
        points[p].x, points[p].y, points[p].value)
  end
end

task main()
  c.printf("Running Index tester...\n")

  var chunks = 4

  var points = region(ispace(ptr,16), point)
  var part = partition(equal, points, ispace(int1d, chunks))

  fill(points.x, 3.0)
  fill(points.y, 5.0)

  __demand(__parallel)
  for i = 0, chunks do
    init_task(part[i])
  end

  calc_task(part[0])

  __demand(__parallel)
  for i = 1, chunks do
    calc_task_left(part[i], part[i-1])
  end

  printer(points)
 -- for i = 0, chunks do
  --end

  c.printf("Done!\n")
end

regentlib.start(main)
