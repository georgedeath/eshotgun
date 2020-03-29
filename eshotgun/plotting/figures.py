import numpy as np
import matplotlib.pyplot as plt
from .. import test_problems
from matplotlib.ticker import StrMethodFormatter


def results_plot_maker(ax, yvals, y_labels, xvals, col_idx,
                       xlabel, ylabel, title, colors,
                       LABEL_FONTSIZE, TITLE_FONTSIZE, TICK_FONTSIZE,
                       semilogy=False, use_fill_between=True):
    # here we assume we're plotting to a matplotlib axis object
    # and yvals is a LIST of arrays of size (n_runs, iterations),
    # where each can be different sized
    # and if xvals is given then len(xvals) == len(yvals)

    # set the labelling
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)

    for c, x, Y, Y_lbl in zip(col_idx, xvals, yvals, y_labels):
        color = colors[c]

        # calculate median run and upper/lower percentils
        bot, mid, top = np.percentile(Y, [25, 50, 75], axis=0)

        if use_fill_between:
            ax.fill_between(x, bot.flat, top.flat, color=color, alpha=0.25)

        ax.plot(x, mid, color=color, label='{:s}'.format(Y_lbl))
        ax.plot(x, bot.flat, '--', color=color, alpha=0.25)
        ax.plot(x, top.flat, '--', color=color, alpha=0.25)

    # set the xlim
    min_x = np.min([np.min(x) for x in xvals])
    max_x = np.max([np.max(x) for x in xvals])
    ax.set_xlim([0, max_x + 1])

    if semilogy:
        ax.semilogy()
    else:
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x: >4.1f}'))

    if title is not None and 'PitzDaily' in title:
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x: >4.2f}'))

    ax.axvline(min_x, linestyle='dashed', color='gray', linewidth=1, alpha=0.5)

    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax.tick_params(axis='both', which='minor', labelsize=TICK_FONTSIZE)

    # set the alignment for outer ticklabels
    ax.set_xticks([0, 50, 100, 150, 200])
    ticklabels = ax.get_xticklabels()
    if len(ticklabels) > 0:
        ticklabels[0].set_ha("left")
        ticklabels[-1].set_ha("right")


def box_plot_maker(a, yvals, y_labels, col_idx, colors, y_axis_label,
                   title, logplot,
                   LABEL_FONTSIZE, TITLE_FONTSIZE, TICK_FONTSIZE):

    data = [Y[:, -1] for Y in yvals]

    medianprops = dict(linestyle='-', color='black')

    bplot = a.boxplot(data, patch_artist=True, medianprops=medianprops)

    if y_labels is not None:
        a.set_xticklabels(y_labels, rotation=90)

    for patch, c in zip(bplot['boxes'], col_idx):
        patch.set(facecolor=colors[c])

    a.set_ylabel(y_axis_label, fontsize=LABEL_FONTSIZE)
    a.set_title(title, fontsize=TITLE_FONTSIZE)

    if logplot:
        a.semilogy()
    else:
        a.yaxis.set_major_formatter(StrMethodFormatter('{x: >4.1f}'))
    a.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    a.tick_params(axis='both', which='minor', labelsize=TICK_FONTSIZE)


def plot_convergence(data, problem_names, problem_names_for_paper,
                     problem_logplot, batch_sizes, method_names,
                     method_names_for_paper, LABEL_FONTSIZE, TITLE_FONTSIZE,
                     LEGEND_FONTSIZE, TICK_FONTSIZE,
                     ylabel=None, save=False):
    for problem_name, paper_problem_name, logplot in zip(problem_names,
                                                         problem_names_for_paper,
                                                         problem_logplot):

        # load the problem
        f_class = getattr(test_problems, problem_name)
        f = f_class()
        dim = f.dim
        N = len(batch_sizes)
        width = (16 / 5) * N
        fig, ax = plt.subplots(1, N, figsize=(width, 2.5), sharex='all',
                               sharey=True)

        for a, batch_size in zip(ax.flat, batch_sizes):
            D = {'yvals': [], 'y_labels': [], 'xvals': [], 'col_idx': []}

            method_to_col = {}
            col_counter = 0

            for method_name, paper_method_name in zip(method_names,
                                                      method_names_for_paper):
                res = data[batch_size][problem_name][method_name]

                # take into account the fact the first 2 * dim points are LHS
                # so we start plotting after it
                xvals = np.arange(0, res.shape[1] - 2 * dim)
                res = res[:, 2 * dim:]

                D['yvals'].append(res)
                D['y_labels'].append(paper_method_name)
                D['xvals'].append(xvals)

                if method_name not in method_to_col:
                    method_to_col[method_name] = col_counter
                    col_counter += 1

                D['col_idx'].append(method_to_col[method_name])

            # create total colour range
            colors = plt.cm.rainbow(np.linspace(0, 1, col_counter))

            xlabel = 'Function Evaluations'
            if batch_size == batch_sizes[0]:
                if ylabel is not None:
                    used_ylabel = ylabel
                else:
                    used_ylabel = 'Distance from Optimum'
            else:
                used_ylabel = None
            ttle = '{:s} (q={:d})'.format(paper_problem_name, batch_size)

            results_plot_maker(a,
                               D['yvals'],
                               D['y_labels'],
                               D['xvals'],
                               D['col_idx'],
                               xlabel=xlabel,
                               ylabel=used_ylabel,
                               title=ttle,
                               colors=colors,
                               LABEL_FONTSIZE=LABEL_FONTSIZE,
                               TITLE_FONTSIZE=TITLE_FONTSIZE,
                               TICK_FONTSIZE=TICK_FONTSIZE,
                               semilogy=logplot,
                               use_fill_between=True)

        plt.subplots_adjust(left=0,
                            right=1,
                            bottom=0,
                            top=1,
                            wspace=0.05,
                            hspace=0.2)

        if save:
            fname = f'convergence_{problem_name:s}.pdf'
            plt.savefig(fname, bbox_inches='tight')
        plt.show()

    # create separate legend image
    fig, ax = plt.subplots(1, 1, figsize=(19, 1), sharey=False)
    results_plot_maker(ax,
                       D['yvals'],
                       D['y_labels'],
                       D['xvals'],
                       D['col_idx'],
                       xlabel=None,
                       ylabel=None,
                       title=None,
                       colors=colors,
                       LABEL_FONTSIZE=LABEL_FONTSIZE,
                       TITLE_FONTSIZE=TITLE_FONTSIZE,
                       TICK_FONTSIZE=TICK_FONTSIZE,
                       semilogy=True,
                       use_fill_between=False)

    legend = plt.legend(loc=3, framealpha=1, frameon=False,
                        fontsize=LEGEND_FONTSIZE,
                        handletextpad=0.35,
                        columnspacing=1,
                        ncol=8)

    # increase legend line widths
    for legobj in legend.legendHandles:
        legobj.set_linewidth(5.0)

    # remove all plotted lines
    for _ in range(len(ax.lines)):
        ax.lines.pop(0)

    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*[bbox.extents + np.array([-5, -5, 5, 5])])
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

    if save:
        fname = f'convergence_LEGEND.pdf'
        fig.savefig(fname, dpi='figure', bbox_inches=bbox)
    plt.show()


def plot_convergence_combined(data, problem_names, problem_names_for_paper,
                              problem_logplot, batch_sizes, method_names,
                              method_names_for_paper, LABEL_FONTSIZE,
                              TITLE_FONTSIZE, TICK_FONTSIZE, save=False):

    width = (16 / 5) * len(batch_sizes)
    height = 3 * len(problem_names)
    fig, ax = plt.subplots(len(problem_names), len(batch_sizes),
                           figsize=(width, height),
                           sharex='all', sharey='row')

    for ax_row, problem_name, paper_problem_name, logplot in zip(ax,
                                                                 problem_names,
                                                                 problem_names_for_paper,
                                                                 problem_logplot):
        for a, batch_size in zip(ax_row, batch_sizes):
            # load the problem
            f_class = getattr(test_problems, problem_name)
            f = f_class()
            dim = f.dim

            D = {'yvals': [], 'y_labels': [], 'xvals': [], 'col_idx': []}

            method_to_col = {}
            col_counter = 0

            for method_name, paper_method_name in zip(method_names,
                                                      method_names_for_paper):
                res = data[batch_size][problem_name][method_name]

                # take into account the fact the first 2 * dim points are LHS
                # so we start plotting after it
                xvals = np.arange(0, res.shape[1] - 2 * dim)
                res = res[:, 2 * dim:]

                D['yvals'].append(res)
                D['y_labels'].append(paper_method_name)
                D['xvals'].append(xvals)

                if method_name not in method_to_col:
                    method_to_col[method_name] = col_counter
                    col_counter += 1

                D['col_idx'].append(method_to_col[method_name])

            # create total colour range
            colors = plt.cm.rainbow(np.linspace(0, 1, col_counter))

            # only the bottom row should have x-axis labels
            if problem_name == problem_names[-1]:
                xlabel = 'Function Evaluations'
            else:
                xlabel = None

            # only the left column should have y-axis labels
            if batch_size == batch_sizes[0]:
                ylabel = 'Distance from Optimum'
            else:
                ylabel = None

            ttle = '{:s} (q={:d})'.format(paper_problem_name, batch_size)

            results_plot_maker(a,
                               D['yvals'],
                               D['y_labels'],
                               D['xvals'],
                               D['col_idx'],
                               xlabel=xlabel,
                               ylabel=ylabel,
                               title=ttle,
                               colors=colors,
                               LABEL_FONTSIZE=LABEL_FONTSIZE,
                               TITLE_FONTSIZE=TITLE_FONTSIZE,
                               TICK_FONTSIZE=TICK_FONTSIZE,
                               semilogy=logplot,
                               use_fill_between=True)

            # ensure labels are all in the same place!
            a.get_yaxis().set_label_coords(-0.18, 0.5)

    plt.subplots_adjust(left=0,
                        right=1,
                        bottom=0,
                        top=1,
                        wspace=0.02,
                        hspace=0.15)

    if save:
        probs = '_'.join(problem_names)
        fname = f'convergence_combined_{probs:s}.pdf'
        plt.savefig(fname, bbox_inches='tight')
    plt.show()


def plot_boxplots(data,
                  budgets,
                  problem_names,
                  problem_names_for_paper,
                  problem_logplot,
                  method_names,
                  method_names_for_paper,
                  LABEL_FONTSIZE,
                  TITLE_FONTSIZE,
                  TICK_FONTSIZE,
                  save=False):

    for problem_name, paper_problem_name, logplot in zip(problem_names,
                                                         problem_names_for_paper,
                                                         problem_logplot):
        # load the problem
        # f_class = getattr(test_problems, problem_name)
        # f = f_class()
        # dim = f.dim

        D = {'yvals': [], 'y_labels': [], 'xvals': [], 'col_idx': []}

        method_to_col = {}
        col_counter = 0

        for method_name, paper_method_name in zip(method_names,
                                                  method_names_for_paper):
            res = data[problem_name][method_name]

            D['yvals'].append(res)
            D['y_labels'].append(paper_method_name)

            if method_name not in method_to_col:
                method_to_col[method_name] = col_counter
                col_counter += 1

            D['col_idx'].append(method_to_col[method_name])

        # create total colour range
        colors = plt.cm.rainbow(np.linspace(0, 1, col_counter))

        # plot!
        fig, ax = plt.subplots(1, 3, figsize=(16, 3), sharey=True)

        for i, (a, budget) in enumerate(zip(ax, budgets)):
            YV = [Y[:, :budget] for Y in D['yvals']]
            title = '{:s} (T={:d})'.format(paper_problem_name, budget)

            y_axis_label = 'Distance from Optimum' if i == 0 else None

            box_plot_maker(a,
                           YV,
                           D['y_labels'],
                           D['col_idx'],
                           colors,
                           y_axis_label,
                           title,
                           logplot,
                           LABEL_FONTSIZE,
                           TITLE_FONTSIZE,
                           TICK_FONTSIZE,
                           )

        plt.subplots_adjust(left=0,  # the left side of the subplots of the figure
                            right=1,   # the right side of the subplots of the figure
                            bottom=0,  # the bottom of the subplots of the figure
                            top=1,     # the top of the subplots of the figure
                            wspace=0.03,  # the amount of width reserved for space between subplots,
                                          # expressed as a fraction of the average axis width
                            hspace=0.16)
        if save:
            fname = f'boxplots_{problem_name:s}.pdf'
            plt.savefig(fname, bbox_inches='tight')

        plt.show()


def plot_egreedy_comparison(data,
                            budgets,
                            problem_names,
                            problem_names_for_paper,
                            problem_logplot,
                            method_names,
                            x_labels,
                            LABEL_FONTSIZE,
                            TITLE_FONTSIZE,
                            TICK_FONTSIZE,
                            save=False):

    for problem_name, paper_problem_name, logplot in zip(problem_names,
                                                         problem_names_for_paper,
                                                         problem_logplot):
        # load the problem
        # f_class = getattr(test_problems, problem_name)
        # f = f_class()
        # dim = f.dim

        D = {'yvals': [], 'y_labels': [], 'xvals': [], 'col_idx': []}

        for method_name in method_names:
            res = data[problem_name][method_name]

            D['yvals'].append(res)

        # plot!
        width = (16 / 3) * len(budgets)
        fig, ax = plt.subplots(1, len(budgets), figsize=(width, 1.75), sharey=True)
        ax = [ax] if not isinstance(ax, list) else ax

        for i, (a, budget) in enumerate(zip(ax, budgets)):
            YV = [Y[:, :budget] for Y in D['yvals']]
            YYV = []
            for Y in YV:
                Y.flat[Y.flat == 0] = 1e-15
                YYV.append(Y)
            YV = YYV
            N = len(method_names)
            # offset indicies for each box location to space them out
            box_inds = np.arange(N)
            c = 0
            for i in range(0, N, 2):
                box_inds[i:i + 2] += c
                c += 1

            medianprops = dict(linestyle='-', color='black')
            bplot = a.boxplot([Y[:, -1] for Y in YV],
                              positions=box_inds,
                              patch_artist=True,
                              medianprops=medianprops)

            a.set_xticks(np.arange(3 * len(x_labels))[::3] + 0.5)
            a.set_xticklabels(x_labels, rotation=0)

            for i, patch in enumerate(bplot['boxes']):
                if i % 2 == 0:
                    patch.set_facecolor('g')
                else:
                    patch.set_facecolor('r')
                    patch.set(hatch='//')

            if budget == budgets[0]:
                a.set_ylabel('Distance from Optimum',
                             fontsize=LABEL_FONTSIZE)

            title = '{:s} (q=10)'.format(paper_problem_name, budget)
            a.set_title(title, fontsize=TITLE_FONTSIZE)

            if logplot:
                a.semilogy()
            else:
                a.yaxis.set_major_formatter(StrMethodFormatter('{x: >4.1f}'))

            a.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
            a.tick_params(axis='both', which='minor', labelsize=TICK_FONTSIZE)

        plt.subplots_adjust(left=0,  # the left side of the subplots of the figure
                            right=1,   # the right side of the subplots of the figure
                            bottom=0,  # the bottom of the subplots of the figure
                            top=1,     # the top of the subplots of the figure
                            wspace=0.03,  # the amount of width reserved for space between subplots,
                                          # expressed as a fraction of the average axis width
                            hspace=0.16)
        if save:
            fname = f'egreedy_compare_{problem_name:s}.pdf'
            plt.savefig(fname, bbox_inches='tight')

        plt.show()
