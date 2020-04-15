import random


# Part - 1
def Dice_Stair_1(total_runs):
    Sixty_Plus = []
    for i in range(total_runs):
        step = 0
        for j in range(250):
            dice = random.randint(1, 6)
            # print(dice)
            if step == 0:
                if dice == 1 or dice == 2:
                    continue
            if dice == 1 or dice == 2:
                step -= 1
            elif dice == 3 or dice == 5 or dice == 4:
                step += 1

            while dice == 6:
                dice = random.randint(1, 6)
                # print(dice)
                if step == 0:
                    if dice == 1 or dice == 2:
                        break
                if dice == 1 or dice == 2:
                    step -= 1
                elif dice == 3 or dice == 5 or dice == 4:
                    step += 1
                print("6 Encountered Re-Rolling", end=', ')
            print("Step = ", step, "Dice = ", dice)
        print("Steps Reached At the end of {} Cycle = {}".format(i, step))
        if step > 60:
            Sixty_Plus.append(1)
        else:
            Sixty_Plus.append(0)
    print("We Moved above Sixty Step = ",
          Sixty_Plus.count(1), " AND ", "We are Less Than Sixty Steps = ", Sixty_Plus.count(0))
    print("Probability ({}) Trials and {} Throws per trial = ".format(total_runs, 250), Sixty_Plus.count(1) / 1000)


# Part - 2
# With Custom Number Of Throws
def Dice_Stair_2(throws, total_runs):
    Sixty_Plus = []
    for i in range(total_runs):
        step = 0
        for j in range(throws):
            dice = random.randint(1, 6)
            # print(dice)
            if step == 0:
                if dice == 1 or dice == 2:
                    continue
            if dice == 1 or dice == 2:
                step -= 1
            elif dice == 3 or dice == 5 or dice == 4:
                step += 1

            while dice == 6:
                dice = random.randint(1, 6)
                # print(dice)
                if step == 0:
                    if dice == 1 or dice == 2:
                        break
                if dice == 1 or dice == 2:
                    step -= 1
                elif dice == 3 or dice == 5 or dice == 4:
                    step += 1
                print("6 Encountered Re-Rolling", end=', ')
            print("Step = ", step, "Dice = ", dice)
        print("Steps Reached At the end of {} Cycle = {}".format(i, step))
        if step > 60:
            Sixty_Plus.append(1)
        else:
            Sixty_Plus.append(0)
    print("We Moved above Sixty Step = ", Sixty_Plus.count(1), " AND ",
          "We are Less Than Sixty Steps = ", Sixty_Plus.count(0))
    print("Probability ({}) Trials and {} Throws per trial = ".format(total_runs, throws), Sixty_Plus.count(1) / 1000)


# Part - 3
# With Probability Distribution
def Dice_Stair_3(throws, weights, total_runs):
    Sixty_Plus = []
    for i in range(total_runs):
        dice_v = []
        step = 0
        for j in range(throws):
            dice = random.choices(range(1, 7), weights=weights)
            dice = dice[0]
            # print(dice)
            if step == 0:
                if dice == 1 or dice == 2:
                    dice_v.append(dice)
                    continue
            if dice == 1 or dice == 2:
                step -= 1
            elif dice == 3 or dice == 5 or dice == 4:
                step += 1

            while dice == 6:
                dice = random.randint(1, 6)
                # print(dice)
                if step == 0:
                    if dice == 1 or dice == 2:
                        break
                if dice == 1 or dice == 2:
                    step -= 1
                elif dice == 3 or dice == 5 or dice == 4:
                    step += 1
                print("6 Encountered Re-Rolling", end=', ')
            print("Step = ", step, "Dice = ", dice)
        print("Steps Reached At the end of {} Cycle = {}".format(i, step))
        if step > 60:
            Sixty_Plus.append(1)
        else:
            Sixty_Plus.append(0)

    print("We Moved above Sixty Step = ", Sixty_Plus.count(1), " AND ",
          "We are Less Than Sixty Steps = ", Sixty_Plus.count(0))
    print("Probability ({}) Trials and {} Throws per trial = ".format(total_runs, throws), Sixty_Plus.count(1) / 1000)


# probabilities = [0.2, 0.3, 0.2, 0.1, 0.1, 0.1]

# Dice_Stair_1(1000)
# Dice_Stair_2(250, 1000)
# Dice_Stair_3(250, probabilities, 1000)


# After applying weights to every number of dice the probability of getting to step more than 60 is reduced
# to 0 in 1000 trial runs with 250 throws of dice in each trial_run
