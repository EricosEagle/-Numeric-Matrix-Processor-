from __future__ import annotations
from copy import deepcopy
from enum import IntEnum
from typing import Iterable, List, Tuple, Union

MAIN_MENU = '''1. Add matrices
2. Multiply matrix by a constant
3. Multiply matrices
4. Transpose matrix
5. Calculate a determinant
6. Inverse matrix
0. Exit'''

TRANSPOSE_MENU = '''1. Main diagonal
2. Side diagonal
3. Vertical line
4. Horizontal line'''


class Axis(IntEnum):
    main_diagonal = 1
    secondary_diagonal = 2
    vertical = 3
    horizontal = 4


class Operation(IntEnum):
    exit = 0
    add = 1
    scale = 2
    multiply = 3
    transpose = 4
    determinant = 5
    inverse = 6


class Matrix:

    def __init__(self, list_: List[List[float]]):
        self._rows = len(list_)
        self._cols = len(list_[0])
        self._matrix = deepcopy(list_)

    def __getitem__(self, key: int) -> List[float]:
        return self._matrix[key]

    def __setitem__(self, key: int, value: List[float]):
        if len(value) != self._cols:
            raise ValueError(f':value: is an invalid length! {len(value)} instead of {self._cols}')
        self._matrix[key] = value

    def __iter__(self) -> Iterable[List[float]]:
        return iter(self._matrix)

    def __len__(self) -> int:
        return len(self._matrix)

    def __str__(self) -> str:
        return '\n'.join(
            [' '.join([str(val) for val in row])
             for row in self._matrix]
        ) + '\n'

    def __add__(self, other: Union[Matrix, List[List[float]]]) -> Matrix:
        if isinstance(other, list):
            other = Matrix(other)
        if not isinstance(other, Matrix):
            return NotImplemented
        if not self.dimension == other.dimension:
            raise ValueError('Cannot add matrices with different dimensions!')
        return Matrix(
            [[s + o for s, o in zip(self[i], other[i])]
             for i in range(len(self._matrix))]
        )

    def __sub__(self, other: Union[Matrix, List[List[float]]]):
        if isinstance(other, list):
            other = Matrix(other)
        try:
            return self.__add__(-other)
        except TypeError:
            return NotImplemented

    def __mul__(self, other: Union[int, float, Matrix, List[List[float]]]) -> Matrix:
        if isinstance(other, (int, float)):
            return Matrix.scale(self, other)
        if isinstance(other, list):
            other = Matrix(other)
        if not isinstance(other, Matrix):
            return NotImplemented
        return Matrix.multiply(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __iadd__(self, other):
        temp = self + other
        self._matrix = temp._matrix
        return self

    def __isub__(self, other):
        temp = self - other
        self._matrix = temp._matrix
        return self

    def __imul__(self, other):
        temp = self * other
        self._matrix = temp._matrix
        return self

    def __neg__(self) -> Matrix:
        return Matrix.scale(self, -1)

    def transpose(self, axis: Union[Axis, int] = Axis.main_diagonal) -> Matrix:
        out = self._matrix
        if axis == Axis.main_diagonal:
            out = [list(col) for col in zip(*out)]
        if axis == Axis.secondary_diagonal:
            out = [list(col[::-1]) for col in zip(*out)][::-1]
        if axis == Axis.vertical:
            out = [row[::-1] for row in out]
        if axis == Axis.horizontal:
            out = out[::-1]
        if out is not self._matrix:  # At least one of the previous if statements were executed
            return Matrix(out)
        raise TypeError('Invalid axis of rotation!')

    def minor(self, i: int = 0, j: int = 0) -> Matrix:
        return Matrix(
            [[self._matrix[k][l] for l in range(self._cols) if l != j]
             for k in range(self._rows) if k != i]
        )

    def cofactor(self, i: int = 0, j: int = 0) -> float:
        return ((-1) ** (i + j)) * Matrix.determinant(self.minor(i, j))

    def adjugate(self) -> Matrix:
        cofactor_matrix = Matrix(
            [[self.cofactor(i, j) for j in range(self._cols)]
             for i in range(self._rows)]
        )
        return cofactor_matrix.transpose()

    def inverse(self) -> Matrix:
        determinant = Matrix.determinant(self)
        if determinant == 0:
            raise ValueError('Matrix is not invertible!')
        return (1 / determinant) * self.adjugate()

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    @property
    def dimension(self) -> Tuple[int, int]:
        return self._rows, self._cols

    @property
    def matrix(self) -> List[List[float]]:
        return self._matrix

    @staticmethod
    def input(dim_message: str = 'Enter size of matrix: ',
              matrix_message: str = 'Enter matrix: ') -> Matrix:
        rows, cols = tuple(int(x) for x in input(dim_message).split())
        matrix = []
        if matrix_message:
            print(matrix_message)
        for _ in range(rows):
            row = [float(val) for val in input().split()]
            if len(row) != cols:
                raise ValueError(f'Invalid row length. ({len(row)} instead of {cols}')
            matrix.append(row)
        return Matrix(matrix)

    @staticmethod
    def scale(matrix: Matrix, scalar: float):
        return Matrix(
            [[scalar * val for val in row]
             for row in matrix._matrix]
        )

    @staticmethod
    def multiply(m1: Matrix, m2: Matrix) -> Matrix:
        if m1._cols != m2._rows:
            raise ValueError(f'Cannot multiply matrices with dimensions {m1.dimension} and {m2.dimension}!')
        m2 = m2.transpose()
        return Matrix(
            [[sum(val1 * val2 for val1, val2 in zip(m1[i], m2[j]))
              for j in range(len(m2._matrix))]
             for i in range(len(m1._matrix))]
        )

    @staticmethod
    def determinant(matrix: Matrix) -> float:
        if matrix._rows != matrix._cols:
            raise ValueError(f'Cannot calculate the determinant of a {matrix.dimension} matrix!')
        if matrix._rows == 1:
            return matrix[0][0]
        if matrix._rows == 2:
            return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
        return sum(matrix[0][j] * matrix.cofactor(0, j) for j in range(matrix._cols))


def two_matrices_input() -> Tuple[Matrix, Matrix]:
    return (Matrix.input('Enter size of first matrix: ', 'Enter first matrix:'),
            Matrix.input('Enter size of second matrix: ', 'Enter second matrix: '))


def addition_input() -> str:
    a, b = two_matrices_input()
    return str(a + b)


def scale_input() -> str:
    matrix = Matrix.input()
    scalar = float(input('Enter constant: '))
    return str(scalar * matrix)


def multiply_input() -> str:
    a, b = two_matrices_input()
    return str(a * b)


def transpose_input() -> str:
    print()
    print(TRANSPOSE_MENU)
    operation = int(input('Your choice: '))
    matrix = Matrix.input()
    return str(matrix.transpose(axis=operation))


def determinant_input() -> str:
    matrix = Matrix.input()
    return str(Matrix.determinant(matrix))


def inverse_input() -> str:
    matrix = Matrix.input()
    return str(matrix.inverse())


FUNCTIONS = [addition_input, scale_input, multiply_input, transpose_input, determinant_input, inverse_input]


def matrix_calculator() -> None:
    print(MAIN_MENU)
    operation = int(input('Your choice: '))
    while operation != Operation.exit:
        try:
            result = FUNCTIONS[operation - 1]()
            print('The result is:')
            print(result)
        except ValueError:
            error_message = 'The operation cannot be performed.' if operation != Operation.inverse \
                else "This matrix doesn't have an inverse."
            print(error_message)
        print(MAIN_MENU)
        operation = int(input('Your choice: '))


if __name__ == '__main__':
    matrix_calculator()
