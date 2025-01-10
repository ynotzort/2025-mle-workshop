from flask import Flask

app = Flask('ping')

# example of the simplest web server
@app.route('/ping', methods=["GET"])
def ping():
    return "PONG"

if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
    

# def debug_decorator(func):
#     def inner(*args, **kwargs):
#         print(f"Calling <{func.__name__}>")
#         result = func(*args, **kwargs)
#         print(f"Result: {result}")
#         return result
#     return inner

# # @debug_decorator
# def function():
#     print("hello world")

# function = debug_decorator(function)

# function()


