using Yao

# calculate crossentropy
function crossentropy(p, q)
    return -sum(p .* log.(q))
end


ent_cx(nbit::Int64, list) = (nbit%2 == 0) ? 
chain(chain(nbit,control(i,i+1=>X) for i in list[1] : 2 : list[end-1] ),
chain(nbit,control(i,i+1=>X) for i in list[2] : 2 : list[end-2] )) : 
    chain(chain(nbit,control(i,i+1=>X) for i in list[1] : 2 : list[nbit-2] ),
          chain(nbit,control(i,i+1=>X) for i in list[2] : 2 : list[nbit-1] ))


rx_layer(nbit::Int64, list) = chain(put(nbit, i => Rx(0)) for i in list)
rz_layer(nbit::Int64, list) = chain(put(nbit, i => Rz(0)) for i in list)
params_layer(nbit::Int64, list) = chain(rx_layer(nbit, list),rz_layer(nbit, list),rx_layer(nbit, list))



# for amplitude encoding
function acc_loss_evaluation_am(circuit::ChainBlock ,reg, y_batch::Matrix{Float64}, batch_size::Int64, mid::Int64)
    res = copy(reg) |> circuit
    q_ = zeros(batch_size,3);
    for i = 1:batch_size
        rdm = density_matrix(viewbatch(res, i), (mid,mid+1))
        q_[i,:] = probs(rdm)[1:3]
        q_[i,:] = q_[i,:]/ norm(q_[i,:])
    end
          
    y_batch = y_batch[:,1:3] ;
    
    pred =  collect( [sortperm(vec(q_[i,:]) )[end] for i in 1:batch_size] )
    y_max = collect( [sortperm(vec(y_batch[i,:]) )[end] for i in 1:batch_size] )
        
    acc = sum(pred .== y_max) / batch_size
    loss = crossentropy(y_batch, q_) / batch_size
    acc, loss
end
    

function fisher_am(circuit::ChainBlock, reg, y_train::Matrix{Float64}, batch_size::Int64, mid::Int64)

    f = zeros(Complex, nparameters(circuit), 1) ;
    res = copy(reg) |> circuit   ;
    q = zeros(batch_size, 3)  ;
    
    for i = 1 : batch_size
        rdm = density_matrix(viewbatch(res, i), (mid,mid+1))
        q[i, :] = probs(rdm)[1:3]
        q[i,:] = q[i,:]/ norm(q[i,:]) 
    end
    
    for i in 1 : batch_size
        g0 = expect'(op0, copy( viewbatch(reg, i) ) => circuit)[2]
        g1 = expect'(op1, copy( viewbatch(reg, i) ) => circuit)[2]
        g2 = expect'(op2, copy( viewbatch(reg, i) ) => circuit)[2]
            
        qg = y_train[i, 1]/q[i,1] * g0 + y_train[i, 2]/q[i,2] * g1 + y_train[i, 3]/q[i,3] * g2 ;
        f = f + qg .* qg ;
    end
    f = real(f / batch_size) ;
    f
end   