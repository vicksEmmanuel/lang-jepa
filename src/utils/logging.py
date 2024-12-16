import os


class CSVLogger:
    def __init__(self, fname, *argv):
        """
        I create a CSV logger that writes header columns defined by `argv`.

        Args:
            fname (str): Path to the CSV file.
            *argv: Tuples of the form (format_str, column_name).
                   For example: ('%d', 'epoch'), ('%.5f', 'loss').

        I open the file and write a header row from column names.
        """
        self.fname = fname
        self.types = []
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(self.fname, "w") as f:
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])
                if i < len(argv):
                    print(v[1], end=",", file=f)
                else:
                    print(v[1], end="\n", file=f)

    def log(self, *argv):
        """
        I log a row into the CSV file.
        Each argument corresponds to the column formatting defined in __init__.
        """
        with open(self.fname, "a") as f:
            for i, (t, val) in enumerate(zip(self.types, argv, strict=False), 1):
                end = "," if i < len(argv) else "\n"
                print(t % val, end=end, file=f)


class AverageMeter:
    """
    I keep track of an average value (e.g., loss) over time.
    I store current value, average, sum, count, max, and min.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        I reset all stored statistics.
        """
        self.val = 0.0
        self.avg = 0.0
        self.max = float("-inf")
        self.min = float("inf")
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        """
        I update the meter with a new value.

        Args:
            val (float): new value to record
            n (int): how many instances this value represents (e.g., batch size)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)
        self.min = min(self.min, val)
