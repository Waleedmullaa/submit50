import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create a new CSP crossword generator.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return a 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]

        for variable, word in assignment.items():
            direction = variable.direction

            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]

        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")

            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont

        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border

        letters = self.letter_grid(assignment)

        img = Image.new(
            "RGBA",
            (
                self.crossword.width * cell_size,
                self.crossword.height * cell_size
            ),
            "black"
        )

        font = ImageFont.truetype(
            "assets/fonts/OpenSans-Regular.ttf",
            80
        )

        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]

                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")

                    if letters[i][j]:
                        bbox = draw.textbbox(
                            (0, 0),
                            letters[i][j],
                            font=font
                        )

                        w = bbox[2] - bbox[0]
                        h = bbox[3] - bbox[1]

                        draw.text(
                            (
                                rect[0][0] + ((interior_size - w) / 2),
                                rect[0][1] + ((interior_size - h) / 2) - 10
                            ),
                            letters[i][j],
                            fill="black",
                            font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce consistency and solve the crossword.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Remove values that do not satisfy unary constraints.
        """
        for var in self.crossword.variables:

            for word in self.domains[var].copy():

                if len(word) != var.length:
                    self.domains[var].remove(word)

    def revise(self, x, y):
        """
        Make x arc consistent with y.

        Return True if the domain of x was changed.
        """
        overlap = self.crossword.overlaps[x, y]

        if overlap is None:
            return False

        x_index, y_index = overlap
        revised = False

        for word_x in self.domains[x].copy():

            found_match = False

            for word_y in self.domains[y]:

                if word_x[x_index] == word_y[y_index]:
                    found_match = True
                    break

            if not found_match:
                self.domains[x].remove(word_x)
                revised = True

        return revised

    def ac3(self, arcs=None):
        """
        Enforce arc consistency using the AC-3 algorithm.
        """
        if arcs is None:
            queue = []

            for x in self.crossword.variables:
                for y in self.crossword.neighbors(x):
                    queue.append((x, y))

        else:
            queue = list(arcs)

        while queue:

            x, y = queue.pop(0)

            if self.revise(x, y):

                if len(self.domains[x]) == 0:
                    return False

                for z in self.crossword.neighbors(x):

                    if z != y:
                        queue.append((z, x))

        return True

    def assignment_complete(self, assignment):
        """
        Check whether every variable has been assigned.
        """
        for var in self.crossword.variables:

            if var not in assignment:
                return False

        return True

    def consistent(self, assignment):
        """
        Check whether an assignment satisfies all constraints.
        """

        # Make sure all words are different.
        words = list(assignment.values())

        if len(words) != len(set(words)):
            return False

        # Check lengths.
        for var, word in assignment.items():

            if len(word) != var.length:
                return False

        # Check overlapping letters.
        for var in assignment:

            for neighbor in self.crossword.neighbors(var):

                if neighbor not in assignment:
                    continue

                overlap = self.crossword.overlaps[var, neighbor]

                if overlap is None:
                    continue

                i, j = overlap

                if assignment[var][i] != assignment[neighbor][j]:
                    return False

        return True

    def order_domain_values(self, var, assignment):
        """
        Return domain values ordered by the least-constraining-value
        heuristic.
        """

        values = []

        for word in self.domains[var]:

            eliminated = 0

            for neighbor in self.crossword.neighbors(var):

                if neighbor in assignment:
                    continue

                overlap = self.crossword.overlaps[var, neighbor]

                if overlap is None:
                    continue

                i, j = overlap

                for neighbor_word in self.domains[neighbor]:

                    if word[i] != neighbor_word[j]:
                        eliminated += 1

            values.append((word, eliminated))

        values.sort(key=lambda item: item[1])

        return [word for word, _ in values]

    def select_unassigned_variable(self, assignment):
        """
        Select an unassigned variable using MRV, then degree heuristic.
        """

        unassigned = []

        for var in self.crossword.variables:

            if var not in assignment:
                unassigned.append(var)

        unassigned.sort(
            key=lambda var: (
                len(self.domains[var]),
                -len(self.crossword.neighbors(var))
            )
        )

        return unassigned[0]

    def backtrack(self, assignment):
        """
        Solve the CSP using backtracking search.
        """

        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)

        for value in self.order_domain_values(var, assignment):

            assignment[var] = value

            if self.consistent(assignment):

                result = self.backtrack(assignment)

                if result is not None:
                    return result

            del assignment[var]

        return None


def main():

    # Check command-line arguments.
    if len(sys.argv) not in [3, 4]:
        sys.exit(
            "Usage: python generate.py structure words [output]"
        )

    # Parse command-line arguments.
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword.
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)

    assignment = creator.solve()

    if assignment is None:
        print("No solution.")

    else:
        creator.print(assignment)

        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
