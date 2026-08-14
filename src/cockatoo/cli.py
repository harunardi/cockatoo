import argparse

from .case import Case

def main():
    parser = argparse.ArgumentParser(
        prog="cockatoo",
        description="Cockatoo multipurpose neutron diffusion tool"
    )

    parser.add_argument(
        "input_file",
        help="Path to the cockatoo input file containing the case definition"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("                Cockatoo multipurpose neutron diffusion tool")
    print("=" * 60)


    print(f"Input file: {args.input_file}")

    # Load the case from the input file
    case = Case.from_file(args.input_file)

    # Run the case
    solution = case.run()

    # print the solution summary
    solution.summary()

if __name__ == "__main__":
    main()
