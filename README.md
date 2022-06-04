# Source Count Estimation 

***
### Build Image

```
docker build -t count . 
```

### Start Container with volume

```
docker run -it -p 5000:5000 -v ${PWD}:/app count 
```
### Website accessed

```
 localhost:5000
```
***