import time
from datetime import datetime
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
                print(f"Making API call for {symbol}: {call_function.__name__}")
                
                # Attempt to make the API call and return the result
                return call_function()
            except TwelveDataError as e:
                error_message = str(e)
                
                # Check if the error message is related to API rate limits
                if "run out of API credits" in error_message or "API limit" in error_message:
                    current_minute = datetime.now().minute
                    print(f"API limit reached. Current minute: {current_minute}. Waiting for the next minute before retrying...")
                    
                    self.wait_for_next_minute(current_minute)
                    continue  # Retry the API call after waiting
                else:
                    raise e  # Raise other types of TwelveDataError
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                raise e

    def wait_for_next_minute(self, current_minute):
        """
        Wait until the minute changes before retrying the API call, with updates every 10 seconds.
        """
        while True:
            now = datetime.now()
            current_second = now.second
            if now.minute != current_minute:
                print(f"New minute {now.minute} reached. Resuming API calls...")
                break
            if current_second % 10 == 0:
                print(f"Still in minute {current_minute}, current second: {current_second}. Waiting...")
            time.sleep(1)
