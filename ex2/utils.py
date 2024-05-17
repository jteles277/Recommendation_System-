def calculate_percentage_error(actual_rating, predicted_rating):
    # Calculate the absolute percentage error
    absolute_error = abs(actual_rating - predicted_rating)
    percentage_error = absolute_error / actual_rating * 100
    
    return percentage_error
