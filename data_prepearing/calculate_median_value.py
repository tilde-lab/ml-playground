import pandas as pd
import statistics


def calc_median_value(data, subject_of_request: int):
    data = data.values.tolist()
    phases = list(set([i[0] for i in data]))

    if subject_of_request == 1:
        result = seebeck_median_value(data, phases)

    return result
def seebeck_median_value(data, phases):
    """
    Calculates the median value of Seebeck coefficient from samples with the same 'phase_id' and
    median value of the coordinates within each 'phase_id'.
    Parameters
    ----------
    data : pandas DataFrame
        DataFrame with next columns: 'phase_id', 'Formula', 'Seebeck coefficient',
        'basis_noneq', 'cell_abc', 'sg_n', 'basis_noneq', 'els_noneq'
    phases : list
        List with set of phases
    """
    new_data_list = []

    for phase in phases:
        seebeck = []
        x_y_z = []
        data_for_phase = [string for string in data if string[0] == phase]
        update_data_for_phase = [row for row in data_for_phase if row[6] != '[]']

        if update_data_for_phase == []:
            continue

        for value in update_data_for_phase:
            seebeck.append(value[2])
            x_y_z.append(value[6])

        # if different number of atoms in structure for specific phase_id,
        # consider median value for those cases of which there are more
        if len(set(len(sample) for sample in x_y_z)) > 1:
            count_len = {}
            for l in set(len(sample) for sample in x_y_z):
                for sample in x_y_z:
                    if len(sample) == l:
                        if str(l) in count_len:
                            count_len[str(l)] = count_len[str(l)] + 1
                        else:
                            count_len[str(l)] = 1

            often_len = int(max(count_len, key=count_len.get))

            update_data_for_phase = []

            # delete rows with different number of atoms compared to most cases
            for value in data_for_phase:
                if len(value[6]) != often_len:
                    continue
                else:
                    update_data_for_phase.append(value)

            # repeat again for update data
            x_y_z = []
            seebeck = []
            for value in update_data_for_phase:
                seebeck.append(value[2])
                x_y_z.append(value[6])

        median_seebeck = statistics.median(seebeck)
        median_x_y_z = []

        for atom in range(len(x_y_z[0])):
            x, y, z = [], [], []
            for sample in x_y_z:
                x.append(sample[atom][0])
                y.append(sample[atom][1])
                z.append(sample[atom][2])
            x_median, y_median, z_median = statistics.median(x), statistics.median(y), statistics.median(z)
            median_x_y_z.append([x_median, y_median, z_median])

        new_data_for_phase = update_data_for_phase[0]
        new_data_for_phase[2] = median_seebeck
        new_data_for_phase[6] = str(median_x_y_z)

        new_data_list.append(new_data_for_phase)

    data = pd.DataFrame(
        new_data_list,
        columns=["phase_id", "Formula", "Seebeck coefficient",
                 "entry", "cell_abc", "sg_n", "basis_noneq", "els_noneq"]
    )

    return data