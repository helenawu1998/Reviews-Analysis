def process_output(predictions, out_file):
    '''
        This function writes predictions to the output file in the correct format
        for submission.
        predictions: 1D array of labels for each input in the test data
        out_file: string name of the output file
    '''
    f = open(out_file, 'w')
    for i, p in enumerate(predictions):
        if i == 0:
            f.write("Id,Prediction\n")
        else:
            f.write(i+1, ",", p, "\n")
    f.close()
    return
