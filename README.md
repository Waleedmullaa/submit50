def load_data(filename):
    """
    Load shopping data from a CSV file and return
    evidence and labels.
    """

    evidence = []
    labels = []

    months = {
        "Jan": 0,
        "Feb": 1,
        "Mar": 2,
        "Apr": 3,
        "May": 4,
        "June": 5,
        "Jul": 6,
        "Aug": 7,
        "Sep": 8,
        "Oct": 9,
        "Nov": 10,
        "Dec": 11
    }

    with open(filename, newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            data = [
                int(row["Administrative"]),
                float(row["Administrative_Duration"]),
                int(row["Informational"]),
                float(row["Informational_Duration"]),
                int(row["ProductRelated"]),
                float(row["ProductRelated_Duration"]),
                float(row["BounceRates"]),
                float(row["ExitRates"]),
                float(row["PageValues"]),
                float(row["SpecialDay"]),
                months[row["Month"]],
                int(row["OperatingSystems"]),
                int(row["Browser"]),
                int(row["Region"]),
                int(row["TrafficType"]),
                1 if row["VisitorType"] == "Returning_Visitor" else 0,
                1 if row["Weekend"] == "TRUE" else 0
            ]

            evidence.append(data)

            if row["Revenue"] == "TRUE":
                labels.append(1)
            else:
                labels.append(0)

    return evidence, labels


def train_model(evidence, labels):
    """
    Train a k-nearest neighbor model with k = 1.
    """

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)

    return model


def evaluate(labels, predictions):
    """
    Evaluate predictions and return sensitivity and specificity.
    """

    positive = 0
    negative = 0
    true_positive = 0
    true_negative = 0

    for actual, predicted in zip(labels, predictions):

        if actual == 1:
            positive += 1

            if predicted == 1:
                true_positive += 1

        else:
            negative += 1

            if predicted == 0:
                true_negative += 1

    sensitivity = true_positive / positive
    specificity = true_negative / negative

    return sensitivity, specificity
