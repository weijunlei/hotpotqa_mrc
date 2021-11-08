import collections


def write_predictions(all_examples, all_features, all_results):
    """Write final predictions to the json file and log-odds of null if needed."""
    # logger.info("Writing predictions to: %s" % (output_prediction_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result[0]] = result
    para_results={}
    sent_results={}
    labels={}
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        id = '_'.join(features[0].unique_id.split('_')[:-1])
        if len(features)>1:
            roll_back=None
            para_result=0
            sent_result=[]
            lbs=[]
            overlap=0
            mask1=0
            for (feature_index, feature) in enumerate(features):
                rs=unique_id_to_result[feature.unique_id]
                if rs[1][0]>para_result:
                    para_result=rs[1][0]
                sr=[]
                lb = []
                mask1+=sum(feature.cls_mask[1:])
                for a,b,c in zip(feature.cls_mask[1:],rs[1][1:],feature.cls_label[1:]):
                    if a!=0:
                        lb.append(c)
                        sr.append(b)
                if roll_back is None:
                    roll_back=0
                    lbs.append(feature.cls_label[0])
                    lbs+=lb
                elif roll_back==1:
                    sent_result[-1]=max(sent_result[-1],sr[0])
                    sr=sr[1:]
                    if lbs[0]==0 and feature.cls_label[0]==1:
                        lbs[0]=1
                    lbs+=lb[1:]
                elif roll_back==2:
                    sent_result[-2] = max(sent_result[-2], sr[0])
                    sent_result[-1] = max(sent_result[-1], sr[1])
                    sr=sr[2:]
                    if lbs[0]==0 and feature.cls_label[0]==1:
                        lbs[0]=1
                    lbs+=lb[2:]
                else:
                    lbs+=lb
                sent_result+=sr
                overlap+=roll_back
                roll_back=feature.roll_back
            para_results[id]=para_result
            sent_results[id]=sent_result
            labels[id]=lbs
            assert len(sent_result)+overlap==mask1
            assert len(lbs) + overlap == mask1+1
        else:
            para_results[id]=unique_id_to_result[features[0].unique_id][1][0]
            sent_result=[]
            lbs=[]
            for ind,(a,b,c) in enumerate(zip(unique_id_to_result[features[0].unique_id][1],features[0].cls_mask,features[0].cls_label)):
                if ind==0:
                    lbs.append(c)
                    continue
                if b!=0:
                    lbs.append(c)
                    sent_result.append(a)
            sent_results[id]=sent_result
            labels[id]=lbs
            assert len(sent_result)==(sum(features[0].cls_mask)-1)
            assert len(lbs) == sum(features[0].cls_mask)

    q_para={}
    for k,v in para_results.items():
        try:
            q_para[k.split('_')[0]][0][int(k.split('_')[1])]=v
        except:
            q_para[k.split('_')[0]]=[[0]*10,[0]*10]
            q_para[k.split('_')[0]][0][int(k.split('_')[1])] = v
    for k,v in labels.items():
        q_para[k.split('_')[0]][1][int(k.split('_')[1])]=v[0]
    recall=count=precision=em=rec=acc=0
    for k,v in q_para.items():
        count+=1
        th=0.5
        p11=p10=p01=p00=0
        # print(th,v[0])
        maxlogit=-100
        maxrs=False
        vmax=max(v[0])
        vmin=min(v[0])
        # maxpara=-1
        for indab,(a,b) in enumerate(zip(v[0],v[1])):
            if a>maxlogit:
                maxlogit=a
                if b==1:
                    maxrs=True
                # maxpara=indab
            a=(a-vmin)/(vmax-vmin)
            a = 1 if a > th else 0
            if a==1 and b==1:
                p11+=1
            elif a==1 and b==0:
                p10+=1
            elif a==0 and b==1:
                p01+=1
            elif a==0 and b==0:
                p00+=1
        if p11+p01!=0:
            recall+=p11/(p11+p01)
        else:
            print('error')
        if p11+p10==0:
            print('error')
        else:
            precision+=p11/(p11+p10)
        if p11==2 and p10==0:
            em+=1
        if p01==0:
            rec+=1
        if maxrs:
            acc+=1
    return acc/count,precision/count,em/count,rec/count