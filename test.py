from fastapi import FastAPI
import uvicorn 
import numpy as np 

app = FastAPI()

# use the post decorator directly below this
'''
    Initialize M and B as np arrays
'''
M = np.random.rand(5,5)
B = np.random.rand(5,5)
def f(x):
    pass
 
#Implement the formula 
#Have two function one using numpy and another not using numpy
#Return 
def f(x):
    y = np.dot(M,x) + B
    return y

#initialize x as a 5 * 5 matrix
x = np.random.rand(5,5)
#Make a call to the function
f(x)

#Recreate the function with the sigmoid Function

if __name__ == "__main__":
    uvicorn.run(app)

'''
    Create a requirements.txt
    Upload to render
'''

