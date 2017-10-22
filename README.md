# DataSense

## Building

```
gradle build
gradle copyDependencies
```

## Running

```
./run.sh
```

Then open a browser to [http://localhost:8111/Client.html](http://localhost:8111/Client.html)
At present I'd reccomend an environment variable *compute_library* be set to openblas. The cuda needs performance for conjugate gradient descent. It works - but too much copying to & from GPU

Drag and drop CSV files to the drop zone to see things work

