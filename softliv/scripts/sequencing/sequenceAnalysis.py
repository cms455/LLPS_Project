import re
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Data import CodonTable
from Bio import AlignIO

def translateSequence(sequence):
    """
    Translates a DNA sequence into an amino acid sequence using the standard DNA codon table.

    Parameters:
    sequence (str): The DNA sequence to translate.

    Returns:
    str: The translated amino acid sequence.
    """

    # Define the standard DNA codon table
    table = CodonTable.unambiguous_dna_by_name["Standard"]

    translatedSeq = ''

    # Iterate over the sequence, three nucleotides at a time
    for i in range(0, len(sequence), 3):
        codon = str(sequence[i:i+3])

        # If the codon contains a gap or is not in the codon table, add a gap to the translated sequence
        if '-' in codon or codon not in table.forward_table:
            translatedSeq += '-'
        else:
            # Translate the codon and add the resulting amino acid to the translated sequence
            translatedSeq += Seq(codon).translate()

    return translatedSeq

def muscle(filePaths):
    """
    Aligns sequences using MUSCLE.

    MUSCLE is a multiple sequence alignment program. It needs to be downloaded and installed separately.
    You can download MUSCLE from https://www.drive5.com/muscle/. After downloading, you should add the
    executable to your system's PATH or specify the path to the executable in the muscleExecutable variable
    in this function.

    The input fasta files are expected to contain DNA sequences, not amino acid sequences.

    This function creates two files in the current directory: 'combined.fa', which contains the sequences to be aligned,
    and 'aligned.fa', which contains the aligned sequences, all in DNA form. If these files already exist, they will be overwritten.

    Parameters:
    filePaths (list): A list of file paths to fasta files. Each element can be a string (the file path) or a tuple
                      (the file path and a string indicating whether to reverse, complement, or reverse_complement
                      the sequence).

    Prints:
    str: The command used to run MUSCLE and any status updates from the command.
    """

    # Read sequences and store initial lengths
    sequences = []
    for path in filePaths:
        if isinstance(path, tuple):
            seq = SeqIO.read(path[0], 'fasta')
            if path[1] == 'reverse':
                print(f"Reversing sequence from file: {path[0]}")
                seq.seq = seq.seq[::-1]
            elif path[1] == 'complement':
                print(f"Complementing sequence from file: {path[0]}")
                seq.seq = seq.seq.complement()
            elif path[1] == 'reverse_complement':
                print(f"Applying reverse complement to sequence from file: {path[0]}")
                seq.seq = seq.seq.reverse_complement()
        else:
            seq = SeqIO.read(path, 'fasta')
        sequences.append(seq)

    # Write sequences to a new file for alignment
    SeqIO.write(sequences, 'combined.fa', 'fasta')

    # Define the MUSCLE executable
    muscleExecutable = "muscle5.1.win64.exe"

    # Run MUSCLE
    muscleCommand = f"{muscleExecutable} -align {'combined.fa'} -output {'aligned.fa'}"
    print(f"Running command: {muscleCommand}")
    result = subprocess.run(muscleCommand, shell=True, capture_output=True)

    if result.returncode != 0:
        # Print each line of the stderr output on a new line if it starts with a timestamp
        print("Command status updates:")
        for line in result.stderr.decode().split('\r'):
            if re.match(r'\d{2}:\d{2}', line.strip()):
                print(line)

    print()

def printAlignmentMetrics(filePaths, translate=True):
    """
    Prints metrics about an alignment, including the percentages of no match, partial match, and complete match,
    the percentage of gaps in the original sequences, and the percentage of complete gaps in the alignment.

    This function expects an 'aligned.fa' file to exist in the current working directory. This file should be
    the output of the muscle function.

    Parameters:
    filePaths (list): A list of file paths to fasta files. Each element can be a string (the file path) or a tuple
                      (the file path and a string indicating whether to reverse, complement, or reverse complement
                      the sequence).
    translate (bool): Whether to translate the sequences in the alignment to amino acids. Defaults to True.

    Prints:
    str: The metrics about the alignment.
    """

    # Initialize counters
    noMatchCount = 0
    partialMatchCount = 0
    completeMatchCount = 0

    # Read the alignment
    alignment = AlignIO.read('aligned.fa', 'fasta')

    if translate:
        # Translate the sequences in the alignment
        for record in alignment:
            record.seq = Seq(translateSequence(record.seq))

    # Calculate the maximum length of the sequences
    maxSeqLength = max(len(record.seq) for record in alignment)

    # Read the fasta names from the original file paths
    sequenceIds = []
    initialSequenceLengths = {}
    gapCountsOriginal = {}

    for path in filePaths:
        if isinstance(path, tuple):
            seq = SeqIO.read(path[0], "fasta")
            if path[1] == 'reverse':
                seq.seq = seq.seq[::-1]
            elif path[1] == 'complement':
                seq.seq = seq.seq.complement()
            elif path[1] == 'reverse_complement':
                seq.seq = seq.seq.reverse_complement()
        else:
            seq = SeqIO.read(path, "fasta")

        sequenceIds.append(seq.id)
        initialSequenceLengths[seq.id] = len(seq.seq)
        gapCountsOriginal[seq.id] = seq.seq.count('-')   
        completeMatchGapCount = 0

    # Count gaps in alignment and update match counts
    for i in range(maxSeqLength):
        chars = [record.seq[i] for record in alignment if i < len(record.seq)]
        uniqueChars = set(chars)
        if len(uniqueChars) == len(chars):
            noMatchCount += 1
        elif len(uniqueChars) == 1:
            completeMatchCount += 1
            # If there is a complete match and it's a gap, increment the counter
            if '-' in uniqueChars:
                completeMatchGapCount += 1
        else:
            partialMatchCount += 1

    # Calculate percentages
    totalLength = maxSeqLength
    noMatchPercentage = (noMatchCount / totalLength) * 100
    partialMatchPercentage = (partialMatchCount / totalLength) * 100
    completeMatchPercentage = (completeMatchCount / totalLength) * 100
    gapPercentagesOriginal = {id: (gapCountsOriginal[id] / initialSequenceLengths[id]) * 100 for id in gapCountsOriginal}
    gapPercentageCompleteMatch = (completeMatchGapCount / totalLength) * 100

    print(f"No match: {noMatchPercentage:.2f}%")
    print(f"Partial match: {partialMatchPercentage:.2f}%")
    print(f"Complete match: {completeMatchPercentage:.2f}%")
    for id in gapPercentagesOriginal:
        print(f"Percentage of gaps in original {id}: {gapPercentagesOriginal[id]:.2f}%")
    print(f"Percentage of complete gaps in alignment: {gapPercentageCompleteMatch:.2f}%")

def determineColor(columnLetters):
    """
    Determines the color for a column in a sequence alignment.

    The color is determined based on the letters in the column. If all letters match, the color is green. In the case of three or more sequences,
    if two letters match, the color is yellow. If no letters match, the color is red. If all characters are blanks, no color is returned.

    Parameters:
    columnLetters (list): A list of letters in a column of a sequence alignment.

    Returns:
    str: The color for the column ('green', 'yellow', or 'red'), or None if all characters are blanks.
    """

    # Filter out blank letters
    nonBlankLetters = [letter for letter in columnLetters if letter != '-']
    uniqueLetters = set(nonBlankLetters)

    # No color should be assigned if all characters are blanks
    if len(nonBlankLetters) == 0:
        return None

    # For two rows, return 'green' if the letters match, 'red' if they don't
    if len(columnLetters) == 2:
        return 'green' if columnLetters[0] == columnLetters[1] else 'red'
    
    # For more than two rows
    mostCommonLetterCount = max(nonBlankLetters.count(letter) for letter in uniqueLetters)
    if mostCommonLetterCount == len(columnLetters):
        return 'green'  # All letters match
    elif mostCommonLetterCount == len(columnLetters) - 1:
        return 'yellow'  # Two letters match
    else:
        return 'red'  # No letters match

def createWedge(matchStart, matchLength, angles, oneAngle, mainCircleRadius, ax, color):
    """
    Creates a wedge on a circular plot to represent a match in a sequence alignment.

    The wedge is created at a specific start angle and end angle, which are determined based on the start and end of the match.
    The radius and width of the wedge are determined based on the radius of the main circle.

    Parameters:
    matchStart (int): The start of the match in the sequence alignment.
    matchLength (int): The length of the match in the sequence alignment.
    angles (list): A list of angles for each position in the sequence alignment.
    oneAngle (float): The angle for one position in the sequence alignment.
    mainCircleRadius (float): The radius of the main circle on the plot.
    ax (matplotlib.axes.Axes): The axes on which to create the wedge.
    color (str): The color of the wedge.

    Returns:
    None
    """

    # Calculate the start and end angles for the wedge
    endAngle = np.degrees(angles[matchStart] + oneAngle)
    startAngle = np.degrees(angles[matchStart + matchLength - 1] - oneAngle)

    # Calculate the radius and width of the wedge
    wedgeRadius = mainCircleRadius - 0.02 * mainCircleRadius
    wedgeWidth = 0.04 * mainCircleRadius

    # Create the wedge
    wedge = patches.Wedge((0.5, 0.5), wedgeRadius, startAngle, endAngle, width=wedgeWidth, edgecolor=None, facecolor=color, alpha=0.5, transform=ax.transAxes)

    # Disable clipping
    wedge.set_clip_on(False)

    # Add the wedge to the axes
    ax.add_patch(wedge)

def circleAlignment(filePaths,translate=True,printFig=True):
    """
    Creates a circular plot of a sequence alignment.

    The sequences are read from the specified file paths and aligned using MUSCLE. The alignment is then plotted on a circular plot,
    with each sequence represented as a circle. Matches in the alignment are represented as wedges on the plot.

    Parameters:
    filePaths (list): A list of file paths to fasta files. Each element can be a string (the file path) or a tuple
                      (the file path and a string indicating whether to reverse, complement, or reverse complement
                      the sequence).
    translate (bool): Whether to translate the sequences in the alignment. Default is True.
    printFig (bool): Whether to print the figure. Default is True.

    Returns:
    None
    """
        
    # Read the alignment
    alignment = AlignIO.read('aligned.fa', 'fasta')

    # Read the fasta names from the original file paths
    sequenceIds = []
    for path in filePaths:
        if isinstance(path, tuple):
            seq = SeqIO.read(path[0], "fasta")
            if path[1] == 'reverse':
                seq.seq = seq.seq[::-1]
            elif path[1] == 'complement':
                seq.seq = seq.seq.complement()
            elif path[1] == 'reverse_complement':
                seq.seq = seq.seq.reverse_complement()
        else:
            seq = SeqIO.read(path, "fasta")

        sequenceIds.append(seq.id)

    # Sort the alignment based on the order of sequenceIds
    alignment.sort(key=lambda record: sequenceIds.index(record.id))

    if translate:
        # Translate the sequences in the alignment
        for record in alignment:
            record.seq = Seq(translateSequence(record.seq))
    
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
    ax.set_aspect('equal')
    ax.axis('off')

    numRows = len(alignment)  # Number of rows is the number of sequences in the alignment

    print(f"Number of rows: {numRows}")

    # Get the maximum label length
    maxLabelLength = max(len(record.id) for record in alignment)

    # Get the sequences from the alignment and prepend and append the padded sequence ID
    sequences = [" " + record.id.ljust(maxLabelLength) + "  " + str(record.seq) + " " for record in alignment[::-1]]

    numLetters = max(len(seq) for seq in sequences)
    mainCircleRadius = ((0.02+0.003*numRows)*numLetters - 0.112) / (2 * np.pi) # crazy formula I figured out using science and big brain

    for row in range(numRows):
        circleRadius = mainCircleRadius + row * 0.03  # Adjust this value to change the distance between the circles
        angles = np.linspace(0, 2 * np.pi, numLetters, endpoint=False)[::-1] + np.pi/2 # Reverse the order of the angles
        textPositions = [(circleRadius * np.cos(angle), circleRadius * np.sin(angle)) for angle in angles]

        for i, pos in enumerate(textPositions):
            angleDegrees = np.degrees(angles[i]) - 90

            # Check if the current character is part of the label or the sequence
            if i < maxLabelLength + 3:  # The "+ 3" accounts for the spaces added to the start and end of the label
                color = 'white'
            else:
                # Check the letters in the column and color them accordingly
                columnLetters = [sequences[r][i] for r in range(numRows)]
                if numRows == 2:  # Special case for two sequences
                    if columnLetters[0] == columnLetters[1]:  # Both letters match
                        color = 'green'
                    else:  # Letters don't match
                        color = 'red'
                else:  # General case for more than two sequences
                    if len(set(columnLetters)) == 1:  # All letters match
                        color = 'green'
                    elif len(set(columnLetters)) == 2:  # Two letters match
                        if columnLetters.count('-') == 2:  # Special case: two blanks and one letter
                            color = 'red'
                        else:
                            color = 'yellow'
                    else:  # No letters match
                        color = 'red'

            ax.text(0.5+pos[0], 0.5+pos[1], sequences[row][i], ha='center', va='center', 
                    fontsize=12, fontname='monospace', rotation=angleDegrees, color=color, transform=ax.transAxes)

    matchStart = None
    prevColor = None
    oneAngle = abs(angles[1] - angles[0])/2

    for i in range(numLetters):
        # Skip label padding
        if i < maxLabelLength + 2:
            continue

        columnLetters = [sequences[r][i] for r in range(numRows)]
        color = determineColor(columnLetters)

        # Skip wedge creation if all are blanks
        if color is None:
            if matchStart is not None:
                # Finish the previous wedge
                createWedge(matchStart, i - matchStart, angles, oneAngle, mainCircleRadius, ax, prevColor)
                matchStart = None
            continue

        # For the wedges, if we're continuing a match of the same color, no action needed
        if matchStart is not None and color == prevColor:
            continue

        # If we're starting a new match or changing color, create the wedge for the previous match
        if matchStart is not None:
            createWedge(matchStart, i - matchStart, angles, oneAngle, mainCircleRadius, ax, prevColor)

        # Start a new match
        matchStart = i
        prevColor = color

    # Create the final wedge if there's an ongoing match at the end
    if matchStart is not None:
        createWedge(matchStart, numLetters - matchStart, angles, oneAngle, mainCircleRadius, ax, prevColor)

    if printFig:
        if translate:
            plt.savefig('alignment_aa.png', dpi=300, bbox_inches='tight', pad_inches=0, facecolor='black')
        else:
            plt.savefig('alignment_dna.png', dpi=300, bbox_inches='tight', pad_inches=0, facecolor='black')

    ax.axis('off')
    plt.show()

def highlightTrypsinSites(fastaFilePath, outputFileName):
    """
    Highlights trypsin cleavage sites in a sequence read from a fasta file.

    Trypsin cleavage sites are where the amino acids lysine (K) or arginine (R) are followed by any amino acid other than proline (P).
    These sites are highlighted in red on a circular plot of the sequence.

    Parameters:
    fastaFilePath (str): The file path to the fasta file.

    Returns:
    None
    """

    # Read the sequence from the fasta file
    record = SeqIO.read(fastaFilePath, "fasta")
    seq = " " + record.id + "  " + str(record.seq) + " "

    # Create a list of colors for each letter in the sequence
    colors = ['white'] * (len(record.id) + 3) + ['red' if (seq[i] in ['K', 'R'] and seq[i+1] != 'P') else 'white' for i in range(len(record.id) + 3, len(seq)-1)] + ['white']

    # Create a matplotlib figure and axes
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
    ax.set_aspect('equal')
    ax.axis('off')

    # Calculate the radius of the main circle for the plot
    numLetters = len(seq)
    mainCircleRadius = ((0.02+0.003)*numLetters - 0.112) / (2 * np.pi)

    # Calculate the positions of the text for each letter in the sequence on the circular plot
    angles = np.linspace(0, 2 * np.pi, numLetters, endpoint=False)[::-1] + np.pi/2
    textPositions = [(mainCircleRadius * np.cos(angle), mainCircleRadius * np.sin(angle)) for angle in angles]

    # Initialize variables for tracking divisions and matches
    divisions = 0
    prevColor = colors[0]
    matchStart = None
    oneAngle = abs(angles[1] - angles[0])/2

    # Iterate over each position, adding the corresponding letter to the plot with the appropriate color and rotation
    for i, pos in enumerate(textPositions):
        angleDegrees = np.degrees(angles[i]) - 90
        ax.text(0.5+pos[0], 0.5+pos[1], seq[i], ha='center', va='center', 
                fontsize=12, fontname='monospace', rotation=angleDegrees, color=colors[i], transform=ax.transAxes)

        # If we're starting a new match or changing color, create the wedge for the previous match
        if matchStart is not None and colors[i] != 'red':
            createWedge(matchStart, i - matchStart, angles, oneAngle, mainCircleRadius, ax, 'red')
            matchStart = None  # Reset matchStart

        # Start a new match
        if colors[i] == 'red' and matchStart is None:
            matchStart = i

        # If the color has changed from 'red' to 'white', increment the division counter
        if prevColor == 'red' and colors[i] == 'white':
            divisions += 1

        # Update the previous color
        prevColor = colors[i]

    # Create the final wedge if there's an ongoing match at the end
    if matchStart is not None:
        createWedge(matchStart, numLetters - matchStart, angles, oneAngle, mainCircleRadius, ax, 'red')

    # Add the number of divisions to the center of the circle
    ax.text(0.5, 0.5, str(divisions), ha='center', va='center', 
            fontsize=100, fontname='monospace', color='red', transform=ax.transAxes)

    # Save the figure to a file and display the plot
    plt.savefig(outputFileName+'.png', dpi=300, bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.show()