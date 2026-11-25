import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from data.dataset import ExDarkDataset


dataset = ExDarkDataset(
    r"D:\12th Trimester\DIP\DIP Paper\Datasets\ExDark\ExDark"
)

sample = dataset[0]

print("=" * 60)
print("DATASET DEBUG")
print("=" * 60)

print("Type:")
print(type(sample))

print()

print("Sample:")
print(sample)

print()

if isinstance(sample, dict):
    print("Dictionary keys:")
    print(sample.keys())

    for key, value in sample.items():
        print()
        print("KEY:", key)
        print("TYPE:", type(value))

        if hasattr(value, "shape"):
            print("SHAPE:", value.shape)

else:
    print("Not a dictionary.")

    if isinstance(sample, (tuple, list)):
        print("Number of elements:", len(sample))

        for i, value in enumerate(sample):
            print()
            print("Element:", i)
            print("TYPE:", type(value))

            if hasattr(value, "shape"):
                print("SHAPE:", value.shape)

print("=" * 60)