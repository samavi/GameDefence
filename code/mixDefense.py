import tensorflow as tf2
import math
def mixedDefense(n,ite, num_poison_points=1):
    #radius is (1 - filter everything)
    #percentile is 1- radius
    learning_rate = 0.05
    print_rate = ite/10
    gap = 0.15/n
    #gap=0.05
    percentileBased = True
    #create initial radius
    radii = []
    ini =0.0
    #initial radius
    for i in range(n):
        ini = ini + gap
        print(ini)
        radii.append(tf2.contrib.eager.Variable(ini))

    for j in range(ite):
        total_gamma = 0
        #prepare computation of gradient
        with tf2.GradientTape() as t:
            t.watch(radii)
            #compute percentage for indifference strategy
            percentage = calcPerccentage(radii, percentileBased)
            p=0
            for i in range (n):
                #compute expected gamma value
                total_gamma = total_gamma +Gamma(radii[i])* percentage[i]
            #the payoff function
            cost = tf2.reduce_mean(num_poison_points * E(radii[n-1]) + total_gamma)
            #print("cost")
            #print(cost)
        #compute gradient of payoff function
        cost_grad = t.gradient(cost, radii)
        #clip gradient
        clipped_grad = tf2.clip_by_value(cost_grad, -1,1)
        #print intermidiate result for testing
        if (j+1) % print_rate == 0:
            print("ite"+str(j))
            print("cost")
            print(cost)
            for i in range(n):
                print(clipped_grad[i].numpy())
                #print(cost_grad[i].numpy())
                print(radii[i])
                #print(cost)
        for i in range(n):
            #gradient descent to adjust each radius
            #restrict radius into 0-100
            #new_rad = tf2.maximum(
            #    radii[i] - learning_rate*clipped_grad[i].numpy(),
            #    0)
            
            new_rad = tf2.minimum(tf2.maximum(
                radii[i] - learning_rate*clipped_grad[i].numpy(),
                0), 1)
            #print("rad")
            #print(new_rad)
            var = tf2.contrib.eager.Variable(new_rad)
            radii[i] = var

    print("Final Result: ")
    print(radii)
    print(percentage)
    return radii, percentage

def calcPerccentage(radii, percentileBased = True):
    #each contains a radius
    #retrieve the ordering of the radii, ascending order
    order = insertion_index(radii)
    #print("Calc")
    #print(radii)
    
    last = radii[order[len(order)-1]]
    #print(last)
    result = [0]*len(order)
    for o in order:
        #percentile based
        if percentileBased:
            temp = E(last)/E(radii[o])
        else:
            #loss based
            temp = E(radii[o])/E(last)
        #print(temp)
        for i in range(len(order)-1):
            temp = temp - result[i]
        result[o] = temp
    #print("abc")
    return result

def E(radius):
    #print(radius)
    assert radius>=0.0
    #assert radius<=1

    

    #l2 for spambase
    #return -0.316021694*radius**(3/2)+0.4635186795*radius-0.188415746*radius**0.5+0.1069928266
    #l2 mnist17
    #return 0.6082554658* radius**3 - 0.75177172 *radius**2+ 0.1331190318* radius + 0.0766191771
    #lb for spambase
    #return (2.064685562*(radius**(-2.221724743))+1.152335605)**(-14.19438786)

    #PCA spambase
    #return -0.3026798131*radius**1.5 + 0.4859745134*radius - 0.2563186244*radius**0.5 + 0.05920533971
    #PCA M17
    #return 0.06148862459*radius**(24.64415334*radius)

    #Curie Spambase
    #return -22.89302461*radius**3 + 10.59317697*radius**2 - 0.9498090684*radius + 0.1517908779
    #Curie M17
    return 0.0004711628356*radius**1.5 + 0.2828033852*radius - 0.3689131052*radius**0.5 + 0.1132015764
def Gamma(radius):
    assert radius>=0.0
    #assert radius<=1

    #l2 for spambase
    #return -0.07573513456*radius**3+0.1066693975*radius**2+0.1613991427*radius+0.00004772662832
    #l2 for mnist17
    #return 0.06546131049* radius**3 - 0.1042498982*radius**2 + 0.2446892381*radius - 0.00001637642514
    #lb for spambase
    #return -1.13229814*10**(-5)*radius+9.312134064*10**(-2)*math.exp(-1.67008191*radius)+3.283296722*10**(-4)

    #PCA spambase
    #return 0.2484920289*radius**3 - 0.2677299132*radius**2 + 0.2581829859*radius + 0.002600091566
    #PCA M17
    #return 0.02440931169*radius**3 - 0.07542474352 *radius**2 + 0.2358321557*radius + 0.01627123538

    #Curie Spambase
    #return (24.5176527*math.exp(-20.92869612*radius) + 1.049457417)**-34.82620859
    #Curie M17
    return 0.2348033752*radius**0.9433875458 + 0.00008504903822
def insertion_index(l):
    temp = l[:]
    result = []

    for i in range(len(temp)):
        count = 0
        index = 0
        while temp[index] is None:
            index = index+1
        m = temp[index]
        
        for t in temp:
            if t is not None and tf2.less(t, m):
                m = t
                index = count
            count = count +1
        result.append(index)
        temp[index] = None
    return result


tf2.enable_eager_execution()
n = 2
mixedDefense(n,3000)

#print(E(0.7))
