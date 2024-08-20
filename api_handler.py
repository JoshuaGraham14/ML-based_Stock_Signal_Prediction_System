import time
from twelvedata import TDClient
from twelvedata.exceptions import TwelveDataError

class APIHandler:
    def __init__(self, api_key):
        """
        Initialize the APIHandler with the provided API key.
        """
        self.api_key = api_key
        self.td_client = TDClient(apikey=self.api_key)

    def make_api_call(self, call_function, symbol):
        """
        Make an API call and handle potential rate limit exceptions.
        """
        while True:
            try:
                # Log the function name and the symbol being used
                print(f"Making API call for {symbol}: {call_function.__name__} ")
                
                # Attempt to make the API call and return the result
                return call_function()
            except TwelveDataError as e:
                error_message = str(e)
                # print(f"Encountered TwelveDataError: {error_message}")
                
                # Check if the error message is related to API rate limits
                if "run out of API credits" in error_message or "API limit" in error_message:
                    print("API limit reached. Waiting for 60 seconds before retrying...")
                    self.wait_with_countdown(60)
                    continue  # Retry the API call after waiting
                else:
                    raise e  # Raise other types of TwelveDataError
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                raise e

    def wait_with_countdown(self, wait_time):
        """
        Wait for the specified time with a countdown, updating every 10 seconds.
        """
        for i in range(wait_time, 0, -10):
            print(f"Waiting for {i} seconds...")
            time.sleep(10)
        print("Resuming API calls...")
