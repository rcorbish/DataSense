# DataSense

## Building

```
gradle copyDependencies
```

for a Docker image ...

```
gradle docker
```


## Running

```
./run.sh
```

Then open a browser to [http://localhost:8111/app](http://localhost:8111/app)
At present I'd recommend an environment variable *compute_library* be set to *openblas*. The cuda needs performance for conjugate gradient descent. It works - but too much copying to & from GPU

Drag and drop CSV files to the drop zone to see things work

