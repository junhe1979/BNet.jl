#----------------------------------------------------------------------------------------------------------
#核心计算
#----------------------------------------------------------------------------------------------------------
module Evo
using Distributed, JLD2, Printf, ProgressBars #格式输出包
#----------------------------------------------------------------------------------------------------------
# transitions btween different states of reacteptor
struct Reaction{F<:Function,V<:Vector{Int64}}
    rate::F #transition rate
    stateInitial::V
    stateFinal::V
end
#----------------------------------------------------------------------------------------------------------
#倾向函数 （核心函数）
function propensity(state::AbstractVector{Int64}, left::Vector{Int64}, numberStates::Int)::Float64
    return prod(binomial(state[i], left[i]) for i in 1:numberStates)
end

#模拟演化
function evolute(reactionsV::Vector{Vector{Reaction}}, Func::Array{Function}, S, keys)

    timeMax, steps, method = keys.timeMax, keys.steps, keys.method
    numberNodes, numberStates = size(S.R[1])
    t = Float64[0.0] # 时间轨迹，t[0]是初始时间
    Ss = [deepcopy(S)] # fractions of states，n[0] is the initial fractions
    t00 = 0.0 #初始时间
    IP = [Float64[] for _ in reactionsV]
    for (ir, reactions) in enumerate(reactionsV)
        IP0 = Vector{Float64}(undef, length(reactions) * numberNodes)
        for (ireact, react) in enumerate(reactions)
            for iN in 1:numberNodes
                IP0[(ireact-1)*numberNodes+iN] = propensity(view(S.R[ir], iN, 1:numberStates), react.stateInitial, numberStates)
            end
        end
        IP[ir] = IP0
    end
    tplot = 0.0 # for saving the results of fractions
    isteps = 0   # 步数
    dt = timeMax / steps #步长
    #以下为演化
    #while isteps<steps && t00<timeMax #满足最大步数或时间则停止
    iter = 0:steps
    if myid() == 2
        iter = ProgressBar(0:steps)
    end
    # 以下为演化过程
    @inbounds @simd for isteps in iter
        if method == "Fixed"
            t00 += dt # 增加步长
            tplot += dt

            for (ir, reactions) in enumerate(reactionsV)
                iN0 = Int64[] # 记录状态变化的节点

                rate0 = Vector{Float64}(undef, numberNodes) # 预分配
                for (ireact, react) in enumerate(reactions)
                    rate0 .= react.rate(S) # 避免临时分配

                    for iN in 1:numberNodes
                        A = rate0[iN] * IP[ir][(ireact-1)*numberNodes+iN] # 计算反应概率
                        if A * dt > rand(Float64)  # 判断在 dt 时间内是否发生
                            if !in(iN, iN0)
                                for inumberStates in 1:numberStates
                                    S.R[ir][iN, inumberStates] += react.stateFinal[inumberStates] - react.stateInitial[inumberStates]
                                end
                                push!(iN0, iN)
                            end
                        end
                    end
                end

                # 改变倾向
                for (ireact, react) in enumerate(reactions)
                    for iN in iN0
                        IP[ir][(ireact-1)*numberNodes+iN] = propensity(view(S.R[ir], iN, 1:numberStates), react.stateInitial, numberStates)
                    end
                end
            end
        end
        #计算连续变化变量，比如钙浓度等具体函数见main.jl
        for func in Func
            func(S, dt)
        end


        #记录数值
        if abs(tplot - timeMax / keys.plotPoints) < timeMax / keys.steps * 0.1
            push!(Ss, deepcopy(S))  #记录状态
            push!(t, t00) #记录时间
            tplot = 0.0

        end
    end
    return t, Ss
end

end
