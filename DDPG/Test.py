            """
            _,_,l = self.events.sample(self.batch_size)
            ls0, la, lr, ls1, ld=map(list,zip(*l))
            ls0= torch.FloatTensor(ls0)
            la = torch.FloatTensor(la)
            lr = torch.FloatTensor(lr)
            ls1= torch.FloatTensor(ls1)
            ld = torch.FloatTensor(ld)

            #############################################################
            with torch.no_grad():
                q_target=self.target.q(ls1,self.target.policy(ls1))
                y=lr+self.gamma*(1-ld)*q_target
            
            q=self.model.q(ls0,la)
            critic_loss=self.loss(q,y)
            logger.direct_write('critic_loss',critic_loss,self.nbEvents)
            self.q_optim.zero_grad()
            critic_loss.backward()
            self.policy_optim.step()
            #############################################################
            actor_loss= - self.model.q(ls0,self.model.policy(ls0)).mean()
            logger.direct_write('actor_loss',actor_loss,self.nbEvents)
            self.policy_optim.zero_grad()
            actor_loss.backward()
            self.policy_optim.step()
            with torch.no_grad():
                for param,param_target in zip(self.model.parameters(),self.target.parameters()):
                    param_target.data.mul_(self.ru)
                    param_target.data.add_((1-self.ru)*param.data)
            """