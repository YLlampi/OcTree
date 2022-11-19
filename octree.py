from __future__ import print_function

try:
    import numpy as np
except ImportError:
    np = None


class OctNode(object):
    
    def __init__(self, position, size, depth, data):
        """
        Los cubos OctNode tienen una posición y un tamaño
        La posición está relacionada con los objetos que contiene el nodo, pero no es la misma.
        Las ramas (o hijos) siguen un patrón predecible para que los accesos sean sencillos.
        Aquí, - significa menos que el 'origen' en esa dimensión, + significa más.
        ramas: 0 1 2 3 4 5 6 7
        x:      - - - - + + + +
        y:      - - + + - - + +
        z:      - + - + - + - +
        """
        self.position = position
        self.size = size
        self.depth = depth

        ## Todos los OctNodes serán nodos hoja al principio
        ## Luego se subdividirán a medida que se agreguen más objetos
        self.isLeafNode = True

        ## almacenar nuestro objeto, típicamente será uno, pero quizás más
        self.data = data

        ## podría también darle algunas ramas de emtpy mientras estamos aquí.
        self.branches = [None, None, None, None, None, None, None, None]

        half = size / 2

        ## Las coordenadas del cubo
        self.lower = (position[0] - half, position[1] - half, position[2] - half)
        self.upper = (position[0] + half, position[1] + half, position[2] + half)

    def __str__(self):
        data_str = u", ".join((str(x) for x in self.data))
        return u"position: {0}, size: {1}, depth: {2} leaf: {3}, data: {4}".format(
            self.position, self.size, self.depth, self.isLeafNode, data_str
        )


class Octree(object):
    """
    El propio octree, que es capaz de añadir y buscar nodos.
    """
    def __init__(self, worldSize, origin=(0, 0, 0), max_type="nodes", max_value=10):
        self.root = OctNode(origin, worldSize, 0, [])
        self.worldSize = worldSize
        self.limit_nodes = (max_type=="nodes")
        self.limit = max_value

    @staticmethod
    def CreateNode(position, size, objects):
        """Esto crea el propio OctNode."""
        return OctNode(position, size, objects)

    def insertNode(self, position, objData=None):
        """
        Añade el objeto dado al octree si es posible
        Parámetros
        ----------
        position : array_like con 3 elementos
            La ubicación espacial del objeto
        objData : opcional
            Los datos a almacenar en esta posición. Por defecto almacena la posición.
            Si el objeto no tiene un atributo de posición, se asume que el objeto
            se asume que la posición es el propio objeto.
        Devuelve
        -------
        nodo : OctNode o None
            El nodo en el que se almacenan los datos o None si está fuera del
            volumen límite del octree.
        """
        if np:
            if np.any(position < self.root.lower):
                return None
            if np.any(position > self.root.upper):
                return None
        else:
            if position < self.root.lower:
                return None
            if position > self.root.upper:
                return None

        if objData is None:
            objData = position

        return self.__insertNode(self.root, self.root.size, self.root, position, objData)

    def __insertNode(self, root, size, parent, position, objData):
        """Versión privada de insertNode() que se llama recursivamente"""
        if root is None:
            pos = parent.position

            ## el desplazamiento es la mitad del tamaño asignado para este nodo
            offset = size / 2

            ## averiguar en qué dirección nos dirigimos
            branch = self.__findBranch(parent, position)

            ## nuevo centro = posición del padre + (dirección de la rama * desplazamiento)
            newCenter = (0, 0, 0)

            if branch == 0:
                newCenter = (pos[0] - offset, pos[1] - offset, pos[2] - offset )
            elif branch == 1:
                newCenter = (pos[0] - offset, pos[1] - offset, pos[2] + offset )
            elif branch == 2:
                newCenter = (pos[0] - offset, pos[1] + offset, pos[2] - offset )
            elif branch == 3:
                newCenter = (pos[0] - offset, pos[1] + offset, pos[2] + offset )
            elif branch == 4:
                newCenter = (pos[0] + offset, pos[1] - offset, pos[2] - offset )
            elif branch == 5:
                newCenter = (pos[0] + offset, pos[1] - offset, pos[2] + offset )
            elif branch == 6:
                newCenter = (pos[0] + offset, pos[1] + offset, pos[2] - offset )
            elif branch == 7:
                newCenter = (pos[0] + offset, pos[1] + offset, pos[2] + offset )

            return OctNode(newCenter, size, parent.depth + 1, [objData])

        #else: no estamos en nuestra posición, pero tampoco en un nodo hoja
        elif (
            not root.isLeafNode
            and
            (
                (np and np.any(root.position != position))
                or
                (root.position != position)
            )
        ):

            # Estamos en un octNode todavía, necesitamos atravesar más
            branch = self.__findBranch(root, position)
            # Encuentra la nueva escala con la que trabajamos
            newSize = root.size / 2
            # Realiza la misma operación en la rama correspondiente de forma recursiva
            root.branches[branch] = self.__insertNode(root.branches[branch], newSize, root, position, objData)

        elif root.isLeafNode:
            if (
                (self.limit_nodes and len(root.data) < self.limit)
                or
                (not self.limit_nodes and root.depth >= self.limit)
            ):
                root.data.append(objData)
                #return root
            else:
                root.data.append(objData)
                objList = root.data
                root.data = None
                root.isLeafNode = False
                newSize = root.size / 2
                for ob in objList:
                    if hasattr(ob, "position"):
                        pos = ob.position
                    else:
                        pos = ob
                    branch = self.__findBranch(root, pos)
                    root.branches[branch] = self.__insertNode(root.branches[branch], newSize, root, pos, ob)
        return root

    def findPosition(self, position):
        """
        Búsqueda básica que encuentra el nodo hoja que contiene la posición especificada
        Devuelve los objetos hijos de la hoja, o Ninguno si la hoja está vacía o no hay ninguno
        """
        if np:
            if np.any(position < self.root.lower):
                return None
            if np.any(position > self.root.upper):
                return None
        else:
            if position < self.root.lower:
                return None
            if position > self.root.upper:
                return None
        return self.__findPosition(self.root, position)

    @staticmethod
    def __findPosition(node, position, count=0, branch=0):
        """Versión privada de findPosition."""
        if node.isLeafNode:
            return node.data
        branch = Octree.__findBranch(node, position)
        child = node.branches[branch]
        if child is None:
            return None
        return Octree.__findPosition(child, position, count + 1, branch)

    @staticmethod
    def __findBranch(root, position):
        """
        función de ayuda
        devuelve un índice correspondiente a una rama
        que apunta en la dirección que queremos ir
        """
        index = 0
        if (position[0] >= root.position[0]):
            index |= 4
        if (position[1] >= root.position[1]):
            index |= 2
        if (position[2] >= root.position[2]):
            index |= 1
        return index

    def iterateDepthFirst(self):
        """Iterar a través del octree en profundidad"""
        gen = self.__iterateDepthFirst(self.root)
        for n in gen:
            yield n

    @staticmethod
    def __iterateDepthFirst(root):
        """Versión privada (estática) de iterateDepthFirst"""

        for branch in root.branches:
            if branch is None:
                continue
            for n in Octree.__iterateDepthFirst(branch):
                yield n
            if branch.isLeafNode:
                yield branch

## ---------------------------------------------------------------------------------------------------##


if __name__ == "__main__":

    import random
    import time

    class TestObject(object):
        """Clase de objeto ficticio para probar"""
        def __init__(self, name, position):
            self.name = name
            self.position = position

        def __str__(self):
            return u"nombre: {0} posicion: {1}".format(self.name, self.position)

    # Número de objetos que pretendemos añadir.
    NUM_TEST_OBJECTS = 2000

    # Número de búsquedas que vamos a probar
    NUM_LOOKUPS = 2000

    # Tamaño que cubre el octree
    WORLD_SIZE = 100.0

    #ORIGIN = (WORLD_SIZE, WORLD_SIZE, WORLD_SIZE)
    ORIGIN = (0, 0, 0)

    # El rango del que se extraen los valores aleatorios
    RAND_RANGE = (-WORLD_SIZE * 0.3, WORLD_SIZE * 0.3)

    # crear objetos de prueba aleatorios
    testObjects = []
    for x in range(NUM_TEST_OBJECTS):
        the_name = "Node__" + str(x)
        the_pos = (
            ORIGIN[0] + random.randrange(*RAND_RANGE),
            ORIGIN[1] + random.randrange(*RAND_RANGE),
            ORIGIN[2] + random.randrange(*RAND_RANGE)
        )
        testObjects.append(TestObject(the_name, the_pos))

    # crear algunas posiciones aleatorias para encontrar también
    findPositions = []
    for x in range(NUM_LOOKUPS):
        the_pos = (
            ORIGIN[0] + random.randrange(*RAND_RANGE),
            ORIGIN[1] + random.randrange(*RAND_RANGE),
            ORIGIN[2] + random.randrange(*RAND_RANGE)
        )
        findPositions.append(the_pos)

    test_trees = (
        ("nodos", 10),
        ("profundidad", 5)
    )

    for tree_params in test_trees:

        # Crear un nuevo octree, tamaño del mundo
        myTree = Octree(
            WORLD_SIZE,
            ORIGIN,
            max_type=tree_params[0],
            max_value=tree_params[1]
        )

        # Insertar algunos objetos al azar y el tiempo
        Start = time.time()
        for testObject in testObjects:
            myTree.insertNode(testObject.position, testObject)
        End = time.time() - Start

        # imprimir algunos resultados.
        print(NUM_TEST_OBJECTS, "Árbol de nodos generado en", End, "Segundos")
        print("Árbol centrado en", ORIGIN, "con tamaño", WORLD_SIZE)
        if myTree.limit_nodes:
            print("Las hojas de los árboles contienen un máximo de", myTree.limit, "objetos cada uno.")
        else:
            print("El árbol tiene una profundidad máxima de", myTree.limit)

        print("La profundidad primero")
        for i, x in enumerate(myTree.iterateDepthFirst()):
            print(i, ":", x)

        ### Pruebas de búsqueda ###

        # La búsqueda de valores fuera del conjunto de valores del octree debería devolver None
        result = myTree.findPosition((
            ORIGIN[0] + WORLD_SIZE * 1.1,
            ORIGIN[1] + WORLD_SIZE,
            ORIGIN[2] + WORLD_SIZE
        ))
        assert(result is None)

        # Busca algunas posiciones al azar y cronometra
        Start = time.time()
        for the_pos in findPositions:
            result = myTree.findPosition(the_pos)

            ##################################################################################
            # Esto demuestra que los resultados son devueltos - pero puede resultar en una impresión grande
            if result is None:
                print("No hay resultados de la prueba en:", the_pos)
            else:
                print("Resultados de la prueba en:", the_pos)
                if result is not None:
                    for i in result:
                        print("    ", i.name, i.position)
                print()
            ##################################################################################

        End = time.time() - Start

        # imprimir algunos resultados.
        print(str(NUM_LOOKUPS), "Las búsquedas realizadas en", End, "Segundos")