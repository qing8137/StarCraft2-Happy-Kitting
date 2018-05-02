#### Actor
class Actor():
    # __init__
    '''
        Setup
        the actor network, actor target network
        soft_update
        train_op => gradient descent
        policy_grads => gradient of policy/action over target_params
    '''
    def __init__(self, sess, action_dim, action_bound, learning_rate, replacement):
        

    # _build_net
    # Build the neural network. 
    # 2 hidden layer, 30 nodes each
    def build_actor(self, s, scope, trainable):

    # learn
    def learn(self, s):
    
    # choose action
    def choose_action(self, s):

#### Critic



class Critic():
    # __init__
    '''
        Setup
        the critic network, critic target network
        soft_update
        target_q
        td_error
        loss
        train_op => gradient descent
        q_grads gradient of q over action
    '''

    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replacement, a, a_):

    # _build_net
    # Build the neural network. 
    # 2 hidden layer, 30 nodes each
    def build_critic(self, s, a, scope, trainable):

    # learn
    def learn(self, s, a, r, s_):

#### Memory

class Memory():
    # __init__
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.counter = 0

    # store_transitions
    def store_transitions(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_)) # stack the information horizontally
        index = self.counter % self.capacity 
        self.data[index, :] = transition
        self.counter += 1

    # randomize_memory
    def randomized_memory(self, n):
        assert self.counter >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]

#### initialization + sess


#### environment